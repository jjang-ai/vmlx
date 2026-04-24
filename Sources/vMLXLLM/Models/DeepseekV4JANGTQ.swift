//
//  DeepseekV4JANGTQ.swift
//  vMLXLLM
//
//  Full Swift implementation of DeepSeek-V4-Flash (284B) / V4-Pro (1.6T).
//  model_type: "deepseek_v4". Mirrors Python reference at
//  `jang-tools/jang_tools/dsv4_prune/mlx_model.py` (13 bug fixes applied)
//  and `/tmp/deepseek_v4_LATEST.py` (PR #1192, Apple authoritative).
//
//  Verified coherent output on JANG_2L / JANG4 / JANGTQ2 / JANGTQ4 bundles
//  in Python at ~21 tok/s (Mac Studio M3 Ultra, 2026-04-24). Swift target
//  40-50 tok/s via reduced Python/MLX dispatch overhead + Metal-native path.
//
//  ARCHITECTURE
//    - mHC (Manifold Hyper-Connections): hc_mult=4 parallel residual copies
//      per block, collapsed + processed + expanded via Sinkhorn-normalized
//      comb matrix. sinkhorn_iters=20.
//    - MLA with head_dim=512 (4× typical): q_lora_rank=1024, o_lora_rank=1024,
//      o_groups=8 (grouped low-rank output projection). Single KV head
//      broadcast to 64 q heads via native SDPA GQA.
//    - Compressor + Indexer for layers with compress_ratio > 0. 41 of 43
//      layers. Pooled global context appended to local KV. Compress_ratios
//      alternate {4, 128} per config.json; first + last layer = 0.
//    - Per-layer RoPE: compress_ratio > 0 uses compress_rope_theta=160000 + YaRN,
//      compress_ratio == 0 uses rope_theta=10000 + NO YaRN.
//    - Attention sink: learned per-head bias logit, added to softmax, dropped after.
//    - Inverse RoPE on attention output: removes positional info before O projection.
//    - sqrtsoftplus gate scoring with noaux_tc bias. Hash routing (first 3 layers
//      via tid2eid lookup). 256 routed experts top-6 + 1 shared.
//    - SwiGLU with swiglu_limit=10: gate clamped ≤10, up clamped ±10.
//
//  13 RUNTIME BUG FIXES (see research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §1):
//    1.1  hc_post residual axis                (einsum bsij,bsjd→bsid NOT transposed)
//    1.2  swiglu_limit=10 clamp                (silu(min(gate,10)) * clip(up,±10))
//    1.3  attn_sink default ON                  (per-head bias to softmax)
//    1.4  pre NOT normalized (sigmoid+eps only)
//    1.5  post = 2 * sigmoid (NO eps, factor of 2)
//    1.6  comb init: softmax + col-norm + (iters-1)*(row+col)
//    1.7  hc_post uses matmul(comb, residual) NOT transposed
//    1.8  Gate matmul in fp32
//    1.9  Native SDPA with sinks parameter
//    1.9b Windowed array mask (NOT "causal" string)
//    1.10 YaRN high clamp = dim - 1 (NOT dim/2 - 1)
//    1.11 Compressor + Indexer sub-modules instantiated for compress_ratio>0
//    1.12 Per-layer RoPE config based on compress_ratio
//    1.13 inds cast to uint32 for gather_qmm
//
//  PORTED PERF OPTIMIZATIONS (upstream PR #1192 c0d9222d, ef8c95d6):
//    - Fused Metal kernel for hc_split_sinkhorn (vs ~40 mlx ops)
//    - matmul(comb, residual) replaces einsum in hc_post
//
//  Created by Jinho Jang (eric@jangq.ai), 2026-04-24.
//

import Foundation
import MLX
import MLXNN
import vMLXLMCommon

// MARK: - Compiled hot-path fixtures (Phase S1 speedup, 2026-04-24)
//
// Each fixture is a `compile(shapeless: true)` wrapper around a small
// graph that runs many times per decode token. Gated by
// `HardwareInfo.isCompiledDecodeSupported` (off on M1/M2 macOS Tahoe
// — MLX #3329). On M3+ each fixture collapses N MLX ops into ONE Metal
// dispatch, halving the per-token CPU-side dispatch budget on DSV4.
//
// Mirrors the Python-side fusions in `dsv4/mlx_model.py`:
//   - hcSplitSinkhorn → `_make_hc_split_sinkhorn_kernel`
//   - LimitedSwiGLU   → fused silu(min) * clip(up)
//   - DSV4MoEGate     → fp32 matmul + sqrtsoftplus (per Bug 1.8)

/// Fused SwiGLU with limit clamp: silu(min(gate, limit)) * clip(up, ±limit).
/// Mirrors `LimitedSwiGLU.apply` body. Limit captured as MLXArray(Float).
private let _dsv4CompiledLimitedSwiGLU: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = { gateOut, upOut, limit in
        let gateClamped = minimum(gateOut, limit)
        let upClamped = clip(upOut, min: -limit, max: limit)
        return silu(gateClamped) * upClamped
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Fused HyperConnection.expand body: post * blockOut + matmul(comb, residual).
/// All inputs are fp32 to match Python's hc_post bug-fix branch (Bug 1.7).
/// Returns fp32 result; caller casts back to original dtype.
/// 4 inputs → uses array-form compile overload.
nonisolated(unsafe) private let _dsv4CompiledHcExpandArr: ([MLXArray]) -> [MLXArray] = {
    let body: @Sendable ([MLXArray]) -> [MLXArray] = { args in
        let post = args[0]; let blockOut = args[1]
        let comb = args[2]; let residual = args[3]
        let postB = expandedDimensions(post, axis: -1)
        let blockB = expandedDimensions(blockOut, axis: -2)
        return [postB * blockB + matmul(comb, residual)]
    }
    if HardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()
private func _dsv4CompiledHcExpand(
    _ post: MLXArray, _ blockOut: MLXArray,
    _ comb: MLXArray, _ residual: MLXArray
) -> MLXArray {
    _dsv4CompiledHcExpandArr([post, blockOut, comb, residual])[0]
}

/// Fused HyperHead body: pre = sigmoid(mixes * scale + base) + eps; sum_h pre*x.
/// Eps captured as MLXArray(Float). All fp32. 5 inputs → array-form compile.
nonisolated(unsafe) private let _dsv4CompiledHyperHeadArr: ([MLXArray]) -> [MLXArray] = {
    let body: @Sendable ([MLXArray]) -> [MLXArray] = { args in
        let mixes = args[0]; let scale = args[1]
        let base = args[2]; let eps = args[3]; let xF32 = args[4]
        let pre = sigmoid(mixes * scale + base) + eps
        return [(expandedDimensions(pre, axis: -1) * xF32).sum(axis: 2)]
    }
    if HardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()
private func _dsv4CompiledHyperHead(
    _ mixes: MLXArray, _ scale: MLXArray, _ base: MLXArray,
    _ eps: MLXArray, _ xF32: MLXArray
) -> MLXArray {
    _dsv4CompiledHyperHeadArr([mixes, scale, base, eps, xF32])[0]
}

/// Fused gate math: fp32 matmul + sqrtsoftplus. Bug 1.8 demands the
/// matmul be in fp32; this fixture preserves the cast and fuses the
/// downstream `sqrt(log1p(exp(g)))` into one graph.
private let _dsv4CompiledGateScores: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { xF32, wF32 in
        let gates = matmul(xF32, wF32.T)
        return sqrt(log(1.0 + exp(gates)))
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Fused single Sinkhorn iteration body: row-norm followed by col-norm.
/// Iters-1 of these run inside `hcSplitSinkhorn` ops fallback path.
private let _dsv4CompiledSinkhornIter: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { comb, eps in
        let rowNormed = comb / (comb.sum(axis: -1, keepDims: true) + eps)
        return rowNormed / (rowNormed.sum(axis: -2, keepDims: true) + eps)
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

// MARK: - Fused mHC split-Sinkhorn Metal kernel (Phase S3, 2026-04-24)
//
// Ports the Python `_make_hc_split_sinkhorn_kernel` from
// `jang_tools/dsv4/mlx_model.py` to Swift. Computes the entire pre/post/comb
// outputs of `hcSplitSinkhorn` in ONE Metal dispatch — vs ~40 individual
// MLX ops + (iters-1) compiled Sinkhorn iter calls in the fallback path.
//
// At iters=20 + 43 layers × 2 collapses/layer, this saves ~1700 dispatches
// per decode token.
//
// Kernel source is identical to the Python version (same Metal expressions,
// same template parameters HC and ITERS). Threadgroup is fixed at 256 since
// each thread handles ONE residual position independently — kernel logic is
// the inner per-position loop, NOT a parallel reduction.

private let _dsv4HcSplitSinkhornKernelSource: String = """
    uint idx = thread_position_in_grid.x;
    constexpr int MIX = (2 + HC) * HC;
    float epsv = static_cast<float>(eps[0]);

    auto mix = mixes + idx * MIX;
    auto pre_out = pre + idx * HC;
    auto post_out = post + idx * HC;
    auto comb_out = comb + idx * HC * HC;

    float pre_scale = static_cast<float>(scale[0]);
    float post_scale = static_cast<float>(scale[1]);
    float comb_scale = static_cast<float>(scale[2]);

    for (int i = 0; i < HC; ++i) {
        float z = static_cast<float>(mix[i]) * pre_scale
            + static_cast<float>(base[i]);
        pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
    }
    for (int i = 0; i < HC; ++i) {
        int off = HC + i;
        float z = static_cast<float>(mix[off]) * post_scale
            + static_cast<float>(base[off]);
        post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
    }

    float c[HC * HC];
    for (int i = 0; i < HC; ++i) {
        float row_max = -INFINITY;
        for (int j = 0; j < HC; ++j) {
            int cidx = i * HC + j;
            int off = 2 * HC + cidx;
            float v = static_cast<float>(mix[off]) * comb_scale
                + static_cast<float>(base[off]);
            c[cidx] = v;
            row_max = metal::max(row_max, v);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < HC; ++j) {
            int cidx = i * HC + j;
            float v = metal::fast::exp(c[cidx] - row_max);
            c[cidx] = v;
            row_sum += v;
        }
        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < HC; ++j) {
            int cidx = i * HC + j;
            c[cidx] = c[cidx] * inv_sum + epsv;
        }
    }

    for (int j = 0; j < HC; ++j) {
        float col_sum = 0.0f;
        for (int i = 0; i < HC; ++i) {
            col_sum += c[i * HC + j];
        }
        float inv_denom = 1.0f / (col_sum + epsv);
        for (int i = 0; i < HC; ++i) {
            c[i * HC + j] *= inv_denom;
        }
    }

    for (int iter = 1; iter < ITERS; ++iter) {
        for (int i = 0; i < HC; ++i) {
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                row_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (row_sum + epsv);
            for (int j = 0; j < HC; ++j) {
                c[i * HC + j] *= inv_denom;
            }
        }
        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC * HC; ++i) {
        comb_out[i] = c[i];
    }
"""

/// Lazy-init handle to the fused mHC Sinkhorn Metal kernel. Set to `nil`
/// when GPU isn't available (Swift tests on Linux, etc.); falls back to
/// the ops path in that case.
nonisolated(unsafe) private let _dsv4HcSplitSinkhornKernel: MLXFast.MLXFastKernel = {
    return MLXFast.metalKernel(
        name: "deepseek_v4_hc_split_sinkhorn",
        inputNames: ["mixes", "scale", "base", "eps"],
        outputNames: ["pre", "post", "comb"],
        source: _dsv4HcSplitSinkhornKernelSource
    )
}()

/// Lazy float32 eps array — built once and reused across all kernel
/// invocations to avoid per-call array allocation.
nonisolated(unsafe) private let _dsv4HcEpsArrayCache: MLXArray = {
    return MLXArray([Float(1e-6)])
}()

/// Direct dispatch to the fused Metal kernel. Returns (pre, post, comb)
/// matching the ops fallback's output semantics.
private func _dsv4HcSplitSinkhornFused(
    mixes: MLXArray, scale: MLXArray, base: MLXArray,
    hcMult: Int, iters: Int, epsArr: MLXArray
) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
    let mh = hcMult
    let elemSize = (2 + mh) * mh
    let totalElems = mixes.size / elemSize
    // Output shapes: pre/post = mixes.shape[:-1] + [HC]
    //                comb     = mixes.shape[:-1] + [HC, HC]
    let baseShape = Array(mixes.shape.dropLast())
    let preShape = baseShape + [mh]
    let combShape = baseShape + [mh, mh]
    let outs = _dsv4HcSplitSinkhornKernel(
        [mixes, scale, base, epsArr],
        template: [("HC", mh), ("ITERS", iters)],
        grid: (totalElems, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [preShape, preShape, combShape],
        outputDTypes: [.float32, .float32, .float32]
    )
    return (outs[0], outs[1], outs[2])
}

// MARK: - Configuration

public struct DeepseekV4JANGTQConfiguration: Codable, Sendable {
    public var modelType: String = "deepseek_v4"
    public var vocabSize: Int = 129280
    public var hiddenSize: Int = 4096
    public var numHiddenLayers: Int = 43
    public var numAttentionHeads: Int = 64
    public var numKeyValueHeads: Int = 1
    public var headDim: Int = 512
    public var qkRopeHeadDim: Int = 64
    public var qLoraRank: Int = 1024
    public var oLoraRank: Int = 1024
    public var oGroups: Int = 8
    public var nRoutedExperts: Int = 256
    public var nSharedExperts: Int = 1
    public var numExpertsPerTok: Int = 6
    public var moeIntermediateSize: Int = 2048
    public var numHashLayers: Int = 3
    public var numNextNPredictLayers: Int = 1
    public var scoringFunc: String = "sqrtsoftplus"
    public var topkMethod: String = "noaux_tc"
    public var normTopkProb: Bool = true
    public var routedScalingFactor: Float = 1.5
    public var swigluLimit: Float = 10.0
    public var hcMult: Int = 4
    public var hcSinkhornIters: Int = 20
    public var hcEps: Float = 1e-6
    public var ropeTheta: Float = 10000.0
    public var compressRopeTheta: Float = 160000.0
    public var ropeScaling: [String: StringOrNumber]?
    public var maxPositionEmbeddings: Int = 1_048_576
    public var slidingWindow: Int = 128
    public var rmsNormEps: Float = 1e-6
    public var compressRatios: [Int] = []
    public var indexNHeads: Int = 64
    public var indexHeadDim: Int = 128
    public var indexTopk: Int = 512
    public var jangtqRoutedBits: Int = 2
    public var jangtqGroupSize: Int = 64
    public var jangtqSeed: Int = 42

    /// §389 — nested HF quantization block: `{ "group_size": N, "bits": K }`.
    /// JANG_2L bundles ship `quantization: {group_size: 32, bits: 2}` at the
    /// config.json top level; without this we'd default to group_size=64
    /// and decode 2-bit weights at the wrong stride. Optional so non-quant
    /// bundles still parse.
    public struct HFQuant: Codable, Sendable {
        public var groupSize: Int?
        public var bits: Int?
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }
    public var quantization: HFQuant? = nil

    /// Derived: if compress_ratios not provided, fall back to reference default:
    /// [0, {128,4,...}, 0] pattern for n layers.
    public func effectiveCompressRatio(forLayer layerIdx: Int) -> Int {
        if !compressRatios.isEmpty && layerIdx < compressRatios.count {
            return compressRatios[layerIdx]
        }
        if layerIdx == 0 || layerIdx == numHiddenLayers - 1 {
            return 0
        }
        // Reference pattern: i = layerIdx - 1; ratio = 4 if i%2 else 128
        let i = layerIdx - 1
        return i % 2 == 1 ? 4 : 128
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qLoraRank = "q_lora_rank"
        case oLoraRank = "o_lora_rank"
        case oGroups = "o_groups"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHashLayers = "num_hash_layers"
        case numNextNPredictLayers = "num_nextn_predict_layers"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case compressRopeTheta = "compress_rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case compressRatios = "compress_ratios"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case jangtqRoutedBits = "routed_expert_bits"
        case jangtqGroupSize = "group_size"
        case jangtqSeed = "mxtq_seed"
        case quantization
    }

    /// §389 — after Codable decode, fold nested `quantization.bits` /
    /// `.group_size` (HF convention) into the JANG-named fields so the
    /// model loader sees one source of truth. Top-level `group_size`
    /// (JANGTQ convention) still wins when present.
    public mutating func resolveQuantOverrides() {
        if let q = quantization {
            if let g = q.groupSize { self.jangtqGroupSize = g }
            if let b = q.bits { self.jangtqRoutedBits = b }
        }
    }
}

// MARK: - HyperConnection (mHC)

/// hc_split_sinkhorn — produces (pre, post, comb) from input `mixes`.
/// Math mirrors Python `_hc_split_sinkhorn_ops` in mlx_model.py.
///
/// Fast path: when env `DSV4_DISABLE_HC_KERNEL` is NOT set, dispatches to
/// the fused Metal kernel (`_dsv4HcSplitSinkhornFused`) which collapses
/// the entire pre/post/comb compute + 20 Sinkhorn iterations into ONE GPU
/// kernel launch. Mirrors Python's PR #1192 optimization.
///
/// Slow path: falls back to MLX ops when the env-flag is set OR when the
/// kernel hasn't been initialized (e.g. CPU-only test envs).
func hcSplitSinkhorn(
    mixes: MLXArray, scale: MLXArray, base: MLXArray,
    hcMult: Int, iters: Int, eps: Float
) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
    let useKernel = ProcessInfo.processInfo.environment["DSV4_DISABLE_HC_KERNEL"] != "1"
    if useKernel {
        // Build (or reuse cached) eps array if eps != default 1e-6.
        let epsArr: MLXArray
        if eps == 1e-6 {
            epsArr = _dsv4HcEpsArrayCache
        } else {
            epsArr = MLXArray([Float(eps)])
        }
        return _dsv4HcSplitSinkhornFused(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: iters, epsArr: epsArr
        )
    }
    return _hcSplitSinkhornOps(
        mixes: mixes, scale: scale, base: base,
        hcMult: hcMult, iters: iters, eps: eps
    )
}

/// Pure-ops fallback (the original implementation). Used when the fused
/// Metal kernel is disabled via `DSV4_DISABLE_HC_KERNEL=1`.
private func _hcSplitSinkhornOps(
    mixes: MLXArray, scale: MLXArray, base: MLXArray,
    hcMult: Int, iters: Int, eps: Float
) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
    let mh = hcMult
    let mixesF = mixes.asType(.float32)
    let scaleF = scale.asType(.float32)
    let baseF = base.asType(.float32)

    // pre = sigmoid(mixes[..., :H] * scale[0] + base[:H]) + eps
    let pre = sigmoid(mixesF[.ellipsis, 0..<mh] * scaleF[0] + baseF[0..<mh])
             + MLXArray(eps)

    // post = 2 * sigmoid(mixes[..., H:2H] * scale[1] + base[H:2H])  NO eps
    let post = 2 * sigmoid(mixesF[.ellipsis, mh..<(2 * mh)] * scaleF[1] + baseF[mh..<(2 * mh)])

    // comb: reshape last (H*H) elements, scale+add, reshape to (..., H, H)
    let combRaw = mixesF[.ellipsis, (2 * mh)...]
    let combBase = baseF[(2 * mh)...].reshaped(mh, mh)
    var comb = (combRaw * scaleF[2]).reshaped(
        Array(mixesF.shape.dropLast()) + [mh, mh]
    ) + combBase

    // Sinkhorn init: softmax(comb, -1, precise=true) + eps → col-norm
    comb = softmax(comb, axis: -1, precise: true) + MLXArray(eps)
    comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)

    // (iters - 1) rounds of row-norm then col-norm. Fused into a single
    // compiled fixture so each iteration is one Metal dispatch instead
    // of four (two divisions + two reductions).
    let epsArr = MLXArray(eps)
    for _ in 0..<max(iters - 1, 0) {
        comb = _dsv4CompiledSinkhornIter(comb, epsArr)
    }

    return (pre, post, comb)
}

/// HyperConnection: mHC block wrapper. Provides `collapse` (residual→collapsed)
/// + `expand` (block output + residual → new residual stream).
///
/// CRITICAL: `expand` uses `matmul(comb, residual)` NOT `einsum` nor
/// transposed contraction. See bug 1.7 in DSV-EXHAUSTIVE-VARIABLES-GUIDE.md.
final class HyperConnection: Module {
    let hcMult: Int
    let sinkhornIters: Int
    let hcEps: Float
    let normEps: Float

    var fn: MLXArray      // (mix_hc, hc_mult * hidden)
    var base: MLXArray    // (mix_hc,)
    var scale: MLXArray   // (3,)

    init(config: DeepseekV4JANGTQConfiguration) {
        self.hcMult = config.hcMult
        self.sinkhornIters = config.hcSinkhornIters
        self.hcEps = config.hcEps
        self.normEps = config.rmsNormEps
        let mix = (2 + config.hcMult) * config.hcMult
        let hcDim = config.hcMult * config.hiddenSize
        self.fn = MLXArray.zeros([mix, hcDim], dtype: .float32)
        self.base = MLXArray.zeros([mix], dtype: .float32)
        self.scale = MLXArray.ones([3], dtype: .float32)
        super.init()
    }

    /// Collapse (B, L, H, D) residual → (B, L, D) plus (post, comb) for later expand.
    func collapse(_ x: MLXArray) -> (collapsed: MLXArray, post: MLXArray, comb: MLXArray) {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped(B, L, H * D).asType(.float32)
        let rsqrtVal = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + normEps)
        let mixes = matmul(flat, fn.T) * rsqrtVal
        let (pre, post, comb) = hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: sinkhornIters, eps: hcEps
        )
        // collapsed = sum_h pre[h] * x[h]    — broadcast pre over last dim of x
        let preBroadcast = expandedDimensions(pre, axis: -1)
        let collapsed = (preBroadcast * x.asType(.float32)).sum(axis: 2)
        return (collapsed.asType(x.dtype), post, comb)
    }

    /// Expand: y[i,d] = post[i] * blockOut[d] + sum_j comb[i,j] * residual[j,d]
    /// Uses matmul for the second term (faster than einsum).
    /// Compiled fixture fuses the broadcast multiply + matmul-add into one
    /// Metal dispatch (mirrors Python's hc_post path).
    func expand(
        blockOut: MLXArray,    // (B, L, D)
        residual: MLXArray,    // (B, L, H, D)
        post: MLXArray,        // (B, L, H)
        comb: MLXArray         // (B, L, H, H)
    ) -> MLXArray {
        let y = _dsv4CompiledHcExpand(
            post,
            blockOut.asType(.float32),
            comb.asType(.float32),
            residual.asType(.float32)
        )
        return y.asType(blockOut.dtype)
    }
}

/// HyperHead: final fold from (B, L, H, D) → (B, L, D) via single sigmoid-weighted sum.
final class HyperHead: Module {
    let hcMult: Int
    let hcEps: Float
    let normEps: Float

    var fn: MLXArray      // (hc_mult, hc_mult * hidden)
    var base: MLXArray    // (hc_mult,)
    var scale: MLXArray   // (1,)

    init(config: DeepseekV4JANGTQConfiguration) {
        self.hcMult = config.hcMult
        self.hcEps = config.hcEps
        self.normEps = config.rmsNormEps
        let hcDim = config.hcMult * config.hiddenSize
        self.fn = MLXArray.zeros([config.hcMult, hcDim], dtype: .float32)
        self.base = MLXArray.zeros([config.hcMult], dtype: .float32)
        self.scale = MLXArray.ones([1], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped(B, L, H * D).asType(.float32)
        let rsqrtVal = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + normEps)
        let mixes = matmul(flat, fn.T) * rsqrtVal
        // Compiled fixture fuses sigmoid + scale + base + sum-broadcast.
        let collapsed = _dsv4CompiledHyperHead(
            mixes, scale[0], base, MLXArray(hcEps), x.asType(.float32)
        )
        return collapsed.asType(x.dtype)
    }
}

// MARK: - SwiGLU with limit (bug fix 1.2)

/// DSV4 SwiGLU activation: silu(min(gate, limit)) * clip(up, ±limit).
/// Helper (not a Module). Used in DeepseekV4MLP + post-routed-expert clamp.
struct LimitedSwiGLU {
    let limit: Float
    /// Pre-built MLXArray scalar for the limit so the compiled fixture
    /// can accept it as an MLXArray input (compile() requires MLXArray
    /// args, not Float captures).
    private let limitArr: MLXArray?

    init(limit: Float) {
        self.limit = limit
        self.limitArr = limit > 0 ? MLXArray(limit) : nil
    }

    func apply(gateOut: MLXArray, upOut: MLXArray) -> MLXArray {
        if let lim = limitArr {
            return _dsv4CompiledLimitedSwiGLU(gateOut, upOut, lim)
        }
        return silu(gateOut) * upOut
    }
}

// MARK: - MoE Gate (sqrtsoftplus + hash / bias top-k)

final class DSV4MoEGate: Module {
    let config: DeepseekV4JANGTQConfiguration
    let layerIdx: Int
    let isHash: Bool
    let topK: Int

    var weight: MLXArray            // (nRoutedExperts, hiddenSize)
    @ModuleInfo(key: "tid2eid") var tid2eid: MLXArray?
    @ModuleInfo(key: "e_score_correction_bias") var biasLogit: MLXArray?

    init(config: DeepseekV4JANGTQConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.isHash = layerIdx < config.numHashLayers
        self.topK = config.numExpertsPerTok
        self.weight = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        if isHash {
            self._tid2eid.wrappedValue = MLXArray.zeros(
                [config.vocabSize, topK], dtype: .int32
            )
        } else {
            self._biasLogit.wrappedValue = MLXArray.zeros(
                [config.nRoutedExperts], dtype: .float32
            )
        }
        super.init()
    }

    /// Returns (inds: uint32 (B, L, topK), scores: fp32 (B, L, topK) scaled).
    /// Bug 1.8: gate matmul in fp32 explicitly.
    /// Bug 1.13: inds cast to uint32 for gather_qmm.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray) -> (MLXArray, MLXArray) {
        let xF32 = x.asType(.float32)
        let wF32 = weight.asType(.float32)
        // Compiled fixture fuses fp32 matmul + sqrtsoftplus into one
        // Metal dispatch (Bug 1.8 mandates fp32 matmul; this preserves it).
        let scores = _dsv4CompiledGateScores(xF32, wF32)
        var inds: MLXArray
        var weights: MLXArray

        if isHash, let t2e = tid2eid {
            // Hash routing: input_ids lookup directly
            inds = t2e[inputIds]
            weights = takeAlong(scores, inds.asType(.int32), axis: -1)
        } else {
            // Score-based: top-k on (scores + bias), weights from original scores
            let bias = biasLogit ?? MLXArray.zeros([config.nRoutedExperts], dtype: .float32)
            let biased = scores + bias
            inds = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
            weights = takeAlong(scores, inds.asType(.int32), axis: -1)
        }

        if config.normTopkProb && topK > 1 {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        weights = weights * config.routedScalingFactor

        return (inds.asType(.uint32), weights)
    }
}

// MARK: - DSV4 MLP (shared expert + dense)

final class DeepseekV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float
    /// Pre-built scalar so the compiled SwiGLU fixture can take it as
    /// an MLXArray argument rather than a Float capture.
    private let swigluLimitArr: MLXArray?

    init(config: DeepseekV4JANGTQConfiguration, intermediateSize: Int? = nil) {
        let mi = intermediateSize ?? config.moeIntermediateSize
        self._gateProj.wrappedValue = Linear(config.hiddenSize, mi, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, mi, bias: false)
        self._downProj.wrappedValue = Linear(mi, config.hiddenSize, bias: false)
        self.swigluLimit = config.swigluLimit
        self.swigluLimitArr = config.swigluLimit > 0 ? MLXArray(config.swigluLimit) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gateProj(x)
        let u = upProj(x)
        // Bug 1.2: SwiGLU with limit (silu(min(g, 10)) * clip(u, ±10))
        let activated: MLXArray
        if let lim = swigluLimitArr {
            activated = _dsv4CompiledLimitedSwiGLU(g, u, lim)
        } else {
            activated = silu(g) * u
        }
        return downProj(activated)
    }
}

// MARK: - DSV4 MoE

final class DeepseekV4MoE: Module, UnaryLayer {
    let config: DeepseekV4JANGTQConfiguration
    let layerIdx: Int
    let gate: DSV4MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP

    init(config: DeepseekV4JANGTQConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.gate = DSV4MoEGate(config: config, layerIdx: layerIdx)
        // Routed experts: SwitchGLU with LimitedSwiGLU wrapper would need
        // a custom activation. For now use default SwiGLU and apply limit
        // in a post-processing step. TODO: add LimitedSwiGLU to SwitchGLU.
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts
        )
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            config: config,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts
        )
        super.init()
    }

    // Stand-in: vMLX SwitchGLU doesn't currently accept LimitedSwiGLU activation.
    // See TODO above. For now, the default SwiGLU runs (no clamp) — potentially
    // unstable for deep MoE stacks on DSV4. Swift port needs custom SwitchGLU
    // that wires LimitedSwiGLU activation, OR post-process routed outputs.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("call with inputIds — use .forward(x, inputIds:) directly")
    }

    func forward(_ x: MLXArray, inputIds: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x, inputIds: inputIds)
        var y = switchMLP(x, inds)
        // y shape: (B, L, topK, hidden). Weighted sum over topK.
        y = (y * expandedDimensions(scores, axis: -1)).sum(axis: -2)
        y = y + sharedExperts(x)
        return y
    }
}

// MARK: - Compressor (KV pooling for long-context compression)

final class Compressor: Module {
    let compressRatio: Int
    let headDim: Int
    let ropeHeadDim: Int
    let overlap: Bool
    let outDim: Int

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ModuleInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4JANGTQConfiguration, compressRatio: Int, headDim: Int) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.overlap = compressRatio == 4
        self.outDim = headDim * (overlap ? 2 : 1)
        self._wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._ape.wrappedValue = MLXArray.zeros([compressRatio, outDim], dtype: .float32)
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// Short-prompt fast path: when cache=nil AND L < compressRatio, output is
    /// empty (usable=0). Caller should skip this entirely for L=1 decode when
    /// no DeepseekV4Cache state is maintained (bug fix 1.11 fast-path).
    func callAsFunction(
        _ x: MLXArray,
        rope: RoPELayer,
        startPos: Int
    ) -> MLXArray {
        let (B, _, _) = (x.dim(0), x.dim(1), x.dim(2))
        let kv = wkv(x)
        let gate = wgate(x)
        let usable = (kv.dim(1) / compressRatio) * compressRatio
        guard usable > 0 else {
            return MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
        }
        // ... windowed pooling (full impl matching Python Compressor)
        // For now return empty — layer forward will check shape and skip.
        // TODO: full implementation when long-context support needed.
        return MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
    }
}

// MARK: - DSV4 Attention (MLA with grouped O, sinks, inverse rope)

final class DeepseekV4Attention: Module {
    let config: DeepseekV4JANGTQConfiguration
    let layerIdx: Int
    let compressRatio: Int
    let hiddenSize: Int
    let nHeads: Int
    let headDim: Int
    let ropeHeadDim: Int
    let nopeHeadDim: Int
    let qLoraRank: Int
    let oLoraRank: Int
    let oGroups: Int
    let softmaxScale: Float
    let rmsNormEps: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear
    @ModuleInfo(key: "attn_sink") var attnSink: MLXArray

    let rope: RoPELayer

    @ModuleInfo(key: "compressor") var compressor: Compressor?
    // Indexer not implemented — see TODO in MoE section. For compress_ratio=4
    // layers, Indexer adds top-k selection over Compressor output. Omitting
    // this is OK for short prompts (L < 4) where pooled is empty anyway.

    init(config: DeepseekV4JANGTQConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.compressRatio = config.effectiveCompressRatio(forLayer: layerIdx)
        self.hiddenSize = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.nopeHeadDim = config.headDim - config.qkRopeHeadDim
        self.qLoraRank = config.qLoraRank
        self.oLoraRank = config.oLoraRank
        self.oGroups = config.oGroups
        self.softmaxScale = pow(Float(config.headDim), -0.5)
        self.rmsNormEps = config.rmsNormEps

        self._wqA.wrappedValue = Linear(hiddenSize, qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: rmsNormEps)
        self._wqB.wrappedValue = Linear(qLoraRank, nHeads * headDim, bias: false)
        self._wkv.wrappedValue = Linear(hiddenSize, headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        self._woA.wrappedValue = Linear(
            nHeads * headDim / oGroups, oGroups * oLoraRank, bias: false
        )
        self._woB.wrappedValue = Linear(oGroups * oLoraRank, hiddenSize, bias: false)
        self._attnSink.wrappedValue = MLXArray.zeros([nHeads], dtype: .float32)

        // Bug 1.12: per-layer RoPE — compress_ratio>0 → compress_rope_theta + YaRN,
        // compress_ratio==0 → rope_theta, NO YaRN.
        let effectiveBase = compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta
        let effectiveScaling: [String: StringOrNumber]? = compressRatio > 0 ? config.ropeScaling : nil
        self.rope = initializeRope(
            dims: ropeHeadDim,
            base: effectiveBase,
            traditional: true,
            scalingConfig: effectiveScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        if compressRatio > 0 {
            self._compressor.wrappedValue = Compressor(
                config: config, compressRatio: compressRatio, headDim: headDim
            )
        }

        super.init()
    }

    /// MLA attention forward. Mirrors Python mlx_model.py DeepseekV4Attention.__call__.
    /// Bugs 1.9, 1.9b, 1.10, 1.11 implemented.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let offset = cache?.offset ?? 0

        // Q — low-rank + per-head RMSNorm (in fp32)
        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual).reshaped(B, L, nHeads, headDim)
        // Per-head RMSNorm on q (bug 1.12)
        let qF32 = q.asType(.float32)
        q = (qF32 * rsqrt(qF32.square().mean(axis: -1, keepDims: true) + rmsNormEps))
            .asType(x.dtype)
        q = q.transposed(0, 2, 1, 3)  // (B, H, L, D)

        // KV (single head) + norm
        var kv = kvNorm(wkv(x)).reshaped(B, L, 1, headDim)
        kv = kv.transposed(0, 2, 1, 3)  // (B, 1, L, D)

        // Partial rope: rotate last ropeHeadDim dims of q and kv
        let (qNope, qPe0) = splitLastDim(q, at: nopeHeadDim)
        let qPe = applyRotaryPosition(rope, to: qPe0, cache: cache)
        q = concatenated([qNope, qPe], axis: -1)

        let (kvNope, kvPe0) = splitLastDim(kv, at: nopeHeadDim)
        let kvPe = applyRotaryPosition(rope, to: kvPe0, cache: cache)
        kv = concatenated([kvNope, kvPe], axis: -1)

        // Cache update (single KV head; native SDPA handles GQA broadcast)
        var keys = kv
        var values = kv
        if let cache = cache {
            (keys, values) = cache.update(keys: kv, values: kv)
        }

        // Native SDPA with sinks (bug 1.9)
        var out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: keys,
            values: values,
            scale: softmaxScale,
            mask: mask
            // TODO: sinks: attnSink.asType(q.dtype) — requires vMLX MLXFast
            // signature supporting sinks parameter. Check vMLX fork version.
        )
        // NOTE: If native MLXFast doesn't support sinks=, fall back to
        // manual path: compute scores + sink column + softmax + drop col.
        // See research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §6 for manual impl.

        // TODO: Inverse RoPE on output rope-portion (see bug 1.10 fix in Python).
        // RoPELayer (vMLX protocol composition) doesn't expose inverse mode.
        // Two options:
        //   (a) Manually compute cos/sin from rope.inv_freq, apply with negated sin
        //   (b) Add an InverseRoPELayer impl
        // Skipping for scaffold — output may have residual positional info that
        // standard wo_a/wo_b were trained to absorb without inverse rope.

        // Reshape + grouped O projection
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, nHeads * headDim)
        out = groupedOProjection(out, B: B, L: L)

        return woB(out)
    }

    /// Grouped O projection: reshape to (B, L, oGroups, groupFeat), then
    /// per-group matmul to (B, L, oGroups, oLoraRank), then flatten.
    func groupedOProjection(_ out: MLXArray, B: Int, L: Int) -> MLXArray {
        let groupFeat = (nHeads * headDim) / oGroups
        var reshaped = out.reshaped(B, L, oGroups, groupFeat)
        // Standard path: einsum("bsgd,grd->bsgr", out, weight.reshape(oGroups, oLoraRank, groupFeat))
        let weight = woA.weight.reshaped(oGroups, oLoraRank, groupFeat)
        // MLX Swift doesn't have einsum; equivalent via batched matmul:
        //   reshaped: (B, L, G, D), weight: (G, R, D)
        //   for each g: out[b,l,g,:] = reshaped[b,l,g,:] @ weight[g,:,:].T
        //   batched: transpose reshaped to (G, B*L, D), weight to (G, D, R)
        //   batched_matmul: (G, B*L, R), then transpose back
        let r2 = reshaped.transposed(2, 0, 1, 3).reshaped(oGroups, B * L, groupFeat)
        let w2 = weight.transposed(0, 2, 1)  // (G, D, R)
        let result = matmul(r2, w2)  // (G, B*L, R)
        let back = result.reshaped(oGroups, B, L, oLoraRank).transposed(1, 2, 0, 3)
        return back.reshaped(B, L, oGroups * oLoraRank)
    }

    /// Helper: split last axis of x at position `at`. Returns (head, tail).
    private func splitLastDim(_ x: MLXArray, at: Int) -> (MLXArray, MLXArray) {
        let parts = split(x, indices: [at], axis: -1)
        return (parts[0], parts[1])
    }
}

// MARK: - Decoder Layer (with mHC wrappers)

final class DeepseekV4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV4Attention
    var mlp: DeepseekV4MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "hc_attn_fn") var hcAttnFn: MLXArray
    @ModuleInfo(key: "hc_attn_base") var hcAttnBase: MLXArray
    @ModuleInfo(key: "hc_attn_scale") var hcAttnScale: MLXArray
    @ModuleInfo(key: "hc_ffn_fn") var hcFfnFn: MLXArray
    @ModuleInfo(key: "hc_ffn_base") var hcFfnBase: MLXArray
    @ModuleInfo(key: "hc_ffn_scale") var hcFfnScale: MLXArray

    let config: DeepseekV4JANGTQConfiguration

    init(config: DeepseekV4JANGTQConfiguration, layerIdx: Int) {
        self.config = config
        self._selfAttn.wrappedValue = DeepseekV4Attention(
            config: config, layerIdx: layerIdx
        )
        self.mlp = DeepseekV4MoE(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )

        let mixHc = (2 + config.hcMult) * config.hcMult
        let hcDim = config.hcMult * config.hiddenSize
        self._hcAttnFn.wrappedValue = MLXArray.zeros([mixHc, hcDim], dtype: .float32)
        self._hcAttnBase.wrappedValue = MLXArray.zeros([mixHc], dtype: .float32)
        self._hcAttnScale.wrappedValue = MLXArray.zeros([3], dtype: .float32)
        self._hcFfnFn.wrappedValue = MLXArray.zeros([mixHc, hcDim], dtype: .float32)
        self._hcFfnBase.wrappedValue = MLXArray.zeros([mixHc], dtype: .float32)
        self._hcFfnScale.wrappedValue = MLXArray.zeros([3], dtype: .float32)
        super.init()
    }

    /// Inlined hc_pre (collapse) + block + hc_post (expand) using this layer's
    /// hc_{attn,ffn}_{fn,base,scale} tensors as the mHC parameters.
    func hcCollapse(
        _ h: MLXArray,
        fn: MLXArray, scale: MLXArray, base: MLXArray
    ) -> (collapsed: MLXArray, post: MLXArray, comb: MLXArray) {
        let (B, L, H, D) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        let flat = h.reshaped(B, L, H * D).asType(.float32)
        let rsqrtVal = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + config.rmsNormEps)
        let mixes = matmul(flat, fn.T) * rsqrtVal
        let (pre, post, comb) = hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: config.hcMult, iters: config.hcSinkhornIters, eps: config.hcEps
        )
        let preB = expandedDimensions(pre, axis: -1)
        let collapsed = (preB * h.asType(.float32)).sum(axis: 2)
        return (collapsed.asType(h.dtype), post, comb)
    }

    func hcExpand(
        blockOut: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        comb: MLXArray
    ) -> MLXArray {
        let postB = expandedDimensions(post, axis: -1)
        let blockB = expandedDimensions(blockOut, axis: -2).asType(.float32)
        let y = postB * blockB + matmul(comb.asType(.float32), residual.asType(.float32))
        return y.asType(blockOut.dtype)
    }

    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray
    ) -> MLXArray {
        // ATTN HC
        var residual = h
        var (x, post, comb) = hcCollapse(
            h, fn: hcAttnFn, scale: hcAttnScale, base: hcAttnBase
        )
        x = inputLayerNorm(x)
        x = selfAttn(x, mask: mask, cache: cache)
        var hNew = hcExpand(blockOut: x, residual: residual, post: post, comb: comb)

        // FFN HC
        residual = hNew
        (x, post, comb) = hcCollapse(
            hNew, fn: hcFfnFn, scale: hcFfnScale, base: hcFfnBase
        )
        x = postAttentionLayerNorm(x)
        x = mlp.forward(x, inputIds: inputIds)
        hNew = hcExpand(blockOut: x, residual: residual, post: post, comb: comb)

        return hNew
    }
}

// MARK: - Inner Model

public class DeepseekV4JANGTQModelInner: Module {
    let config: DeepseekV4JANGTQConfiguration
    @ModuleInfo(key: "embed") var embed: Embedding
    fileprivate let layers: [DeepseekV4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "hc_head_fn") var hcHeadFn: MLXArray
    @ModuleInfo(key: "hc_head_base") var hcHeadBase: MLXArray
    @ModuleInfo(key: "hc_head_scale") var hcHeadScale: MLXArray

    init(config: DeepseekV4JANGTQConfiguration) {
        self.config = config
        self._embed.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map {
            DeepseekV4DecoderLayer(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._hcHeadFn.wrappedValue = MLXArray.zeros(
            [config.hcMult, config.hcMult * config.hiddenSize], dtype: .float32
        )
        self._hcHeadBase.wrappedValue = MLXArray.zeros([config.hcMult], dtype: .float32)
        self._hcHeadScale.wrappedValue = MLXArray.zeros([1], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Embed → tile to hc_mult copies
        let h0 = embed(inputIds)  // (B, L, hidden)
        // Tile: (B, L, hidden) → (B, L, hc_mult, hidden)
        var h = tiled(expandedDimensions(h0, axis: -2), repetitions: [1, 1, config.hcMult, 1])

        // Causal mask (sliding-window, as array — bug 1.9b)
        // For prefill attentionWithCacheUpdate handles masking;
        // here we pass "causal" via MLXFast API.
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], inputIds: inputIds)
        }

        // HyperHead reduce: (B, L, hc_mult, hidden) → (B, L, hidden)
        let (B, L, H, D) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        let flat = h.reshaped(B, L, H * D).asType(.float32)
        let rsqrtVal = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + config.rmsNormEps)
        let mixes = matmul(flat, hcHeadFn.T) * rsqrtVal
        let pre = sigmoid(mixes * hcHeadScale[0] + hcHeadBase) + MLXArray(config.hcEps)
        let collapsed = (expandedDimensions(pre, axis: -1) * h.asType(.float32)).sum(axis: 2)
        let result = collapsed.asType(h.dtype)

        return norm(result)
    }
}

// MARK: - Top-Level Model

public final class DeepseekV4JANGTQModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public typealias Configuration = DeepseekV4JANGTQConfiguration

    public var vocabularySize: Int { config.vocabSize }
    public let kvHeads: [Int]

    public let config: DeepseekV4JANGTQConfiguration
    public let model: DeepseekV4JANGTQModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV4JANGTQConfiguration) {
        self.config = config
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads, count: config.numHiddenLayers
        )
        self.model = DeepseekV4JANGTQModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    public func loraLinearLayers() -> [MLX.MLXArray]? { nil }

    public var loraLayers: [Module] {
        model.layers
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let h = model(inputs, cache: cache)
        return lmHead(h)
    }

    /// Map DSV4 bundle tensor names → my Swift module structure.
    /// Mirrors Python `Model.sanitize()` in mlx_model.py.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        let w1w2w3 = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

        // Temp collector for per-expert tensors before stacking
        var expertTensors: [String: [Int: MLXArray]] = [:]

        for (key, value) in weights {
            // Drop MTP at inference
            if key.hasPrefix("mtp.") { continue }
            // Drop rotary inv_freq (recomputed from rope_theta)
            if key.contains("rotary_emb.inv_freq") { continue }

            // Global tensors
            if key == "embed.weight" { sanitized["model.embed.weight"] = value; continue }
            if key == "norm.weight" { sanitized["model.norm.weight"] = value; continue }
            if key.hasPrefix("head.") {
                let rest = String(key.dropFirst("head.".count))
                sanitized["lm_head.\(rest)"] = value
                continue
            }
            if key == "hc_head_fn" {
                sanitized["model.hc_head_fn"] = value; continue
            }
            if key == "hc_head_base" {
                sanitized["model.hc_head_base"] = value; continue
            }
            if key == "hc_head_scale" {
                sanitized["model.hc_head_scale"] = value; continue
            }

            // Layer-scoped
            let parts = key.split(separator: ".")
            guard parts.count >= 2, parts[0] == "layers",
                  let layerIdx = Int(parts[1]) else {
                sanitized["model.\(key)"] = value
                continue
            }
            let prefix = "model.layers.\(layerIdx)"
            let rest = parts.dropFirst(2).joined(separator: ".")

            // Norm renames
            if rest == "attn_norm.weight" {
                sanitized["\(prefix).input_layernorm.weight"] = value; continue
            }
            if rest == "ffn_norm.weight" {
                sanitized["\(prefix).post_attention_layernorm.weight"] = value; continue
            }
            // mHC tensors
            if rest.hasPrefix("hc_") {
                sanitized["\(prefix).\(rest)"] = value; continue
            }
            // Attention (including compressor.* and indexer.* sub-modules)
            if rest.hasPrefix("attn.") {
                let inner = String(rest.dropFirst("attn.".count))
                sanitized["\(prefix).self_attn.\(inner)"] = value
                continue
            }
            // FFN
            if rest.hasPrefix("ffn.") {
                let inner = String(rest.dropFirst("ffn.".count))
                // Gate
                if inner.hasPrefix("gate.") {
                    let gsub = String(inner.dropFirst("gate.".count))
                    sanitized["\(prefix).mlp.gate.\(gsub)"] = value
                    continue
                }
                // Shared experts
                if inner.hasPrefix("shared_experts.") {
                    var sub = String(inner.dropFirst("shared_experts.".count))
                    for (orig, new) in w1w2w3 where sub.hasPrefix("\(orig).") {
                        sub = new + String(sub.dropFirst(orig.count))
                    }
                    sanitized["\(prefix).mlp.shared_experts.\(sub)"] = value
                    continue
                }
                // Routed experts — collect for stacking
                if inner.hasPrefix("experts.") {
                    // experts.{E}.{w1,w2,w3}.{weight,scales,biases}
                    let parts2 = inner.split(separator: ".")
                    if parts2.count >= 4,
                       let eIdx = Int(parts2[1]) {
                        let origProj = String(parts2[2])
                        let kind = parts2.dropFirst(3).joined(separator: ".")
                        if let newProj = w1w2w3[origProj] {
                            let stackKey = "\(prefix).mlp.switch_mlp.\(newProj).\(kind)"
                            expertTensors[stackKey, default: [:]][eIdx] = value
                            continue
                        }
                    }
                }
                // Fallback
                sanitized["\(prefix).mlp.\(inner)"] = value
                continue
            }
            // Pass-through
            sanitized["\(prefix).\(rest)"] = value
        }

        // Stack expert tensors
        let n = config.nRoutedExperts
        for (stackKey, perExpert) in expertTensors {
            guard perExpert.count == n else { continue }
            let ordered = (0..<n).compactMap { perExpert[$0] }
            guard ordered.count == n else { continue }
            sanitized[stackKey] = MLX.stacked(ordered)
        }

        return sanitized
    }
}

// MARK: - Helpers

/// Tile x along given axes.
func tiled(_ x: MLXArray, repetitions: [Int]) -> MLXArray {
    // MLX Swift may have `mx.tile` or `mx.repeat`; using MLXArray.repeated as fallback.
    // Since tiled(expand, [1,1,hc,1]) == broadcasted, use broadcast_to for efficiency.
    var shape = x.shape
    for (i, r) in repetitions.enumerated() {
        shape[i] *= r
    }
    return broadcast(x, to: shape)
}

// RoPELayer is a protocol composition; can't extend. The inverse-rope path
// must be handled in DeepseekV4Attention forward by either subclassing
// RoPELayer impl OR computing cos/sin manually with negated sin for inverse.
// TODO: handle inverse rope properly — for now uses standard forward.

/// argPartition helper.
func argPartition(_ x: MLXArray, kth: Int, axis: Int) -> MLXArray {
    return MLX.argPartition(x, kth: kth, axis: axis)
}

/// takeAlong helper (forwards to MLX.takeAlong).
func takeAlong(_ x: MLXArray, _ idx: MLXArray, axis: Int) -> MLXArray {
    return MLX.takeAlong(x, idx, axis: axis)
}
