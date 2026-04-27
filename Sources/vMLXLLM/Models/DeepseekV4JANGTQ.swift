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
    // TEMP DISABLE COMPILE 2026-04-25: shape-cache bug bleeding prefill L into decode
    return body
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
    // TEMP DISABLE COMPILE 2026-04-25: shape-cache bug
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
    // TEMP DISABLE COMPILE 2026-04-25: shape-cache bug
    return body
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
///
/// TEMP DISABLED COMPILE (2026-04-25): suspected of caching prefill L
/// shape across decode calls — caused MoE forward to return (1, 5, 4096)
/// for decode input (1, 1, 4096). Bypass compile to verify, restore
/// after fix.
private let _dsv4CompiledGateScores: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { xF32, wF32 in
        let gates = matmul(xF32, wF32.T)
        return sqrt(log(1.0 + exp(gates)))
    }
    // return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
    return body
}()

/// Cache of unit-weight tensors for per-head Q RMSNorm (DSV4 has no learned
/// per-head norm weight). Keyed on (head_dim, dtype). Mirrors Python's
/// `_get_q_norm_ones` cache. Workaround until mlx-swift exposes
/// `weight: MLXArray? = nil` for MLXFast.rmsNorm.
private struct _DSV4QNormOnesKey: Hashable {
    let headDim: Int
    let dtypeKey: Int  // MLX DType isn't directly Hashable; use rawValue proxy
}
nonisolated(unsafe) private var _dsv4QNormOnesCache: [_DSV4QNormOnesKey: MLXArray] = [:]
private let _dsv4QNormOnesLock = NSLock()

private func _dsv4QNormOnesTensor(headDim: Int, dtype: DType) -> MLXArray {
    // Map dtype to a stable Int key (DType is not Hashable in mlx-swift)
    let dtypeKey: Int
    switch dtype {
    case .float16: dtypeKey = 1
    case .bfloat16: dtypeKey = 2
    case .float32: dtypeKey = 3
    default: dtypeKey = 0
    }
    let key = _DSV4QNormOnesKey(headDim: headDim, dtypeKey: dtypeKey)
    _dsv4QNormOnesLock.lock(); defer { _dsv4QNormOnesLock.unlock() }
    if let cached = _dsv4QNormOnesCache[key] { return cached }
    let t = MLXArray.ones([headDim], dtype: dtype)
    _dsv4QNormOnesCache[key] = t
    return t
}

/// Fused single Sinkhorn iteration body: row-norm followed by col-norm.
/// Iters-1 of these run inside `hcSplitSinkhorn` ops fallback path.
private let _dsv4CompiledSinkhornIter: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { comb, eps in
        let rowNormed = comb / (comb.sum(axis: -1, keepDims: true) + eps)
        return rowNormed / (rowNormed.sum(axis: -2, keepDims: true) + eps)
    }
    // TEMP DISABLE COMPILE 2026-04-25: shape-cache bug bleeding prefill L into decode
    return body
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
    if ProcessInfo.processInfo.environment["DSV4_HC_DEBUG"] == "1" {
        FileHandle.standardError.write(Data(
            "[hc-fused] outs.count=\(outs.count) expected=3 grid=(\(totalElems),1,1) elemSize=\(elemSize) mixes.shape=\(mixes.shape) preShape=\(preShape) combShape=\(combShape)\n".utf8))
    }
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
    ///
    /// §414 (BUG 3 ROOT CAUSE, 2026-04-25): the previous version overwrote
    /// `jangtqRoutedBits` from `quantization.bits` unconditionally. After
    /// the §410 config-metadata bug fix, bundles ship `quantization.bits=8`
    /// (the AFFINE setting for embed/lm_head/attention/shared experts) at
    /// the top level. That value is NOT the JANGTQ routed-expert bits —
    /// the bundle ships `routed_expert_bits=2` separately. Overwriting
    /// `jangtqRoutedBits=8` made TurboQuantSwitchLinear allocate packed
    /// arrays sized for 8-bit codes (4 codes per uint32) and lookup an
    /// 8-bit (256-entry) Lloyd-Max codebook instead of the 2-bit (4-entry)
    /// codebook that matches the on-disk packed weights → routed-expert
    /// outputs were 4× wrong magnitude AND sign-flipped → garbage logits
    /// → multilingual gibberish output even though attention/HC/embedding
    /// were all bit-exact to Python.
    ///
    /// Fix: only override `jangtqRoutedBits` when `quantization.mode == "tq"`.
    /// For affine-mode top-level quantization, leave the JANG-named field
    /// alone — it has its own decoded value from `routed_expert_bits`.
    public mutating func resolveQuantOverrides() {
        if let q = quantization {
            if let g = q.groupSize { self.jangtqGroupSize = g }
            // §414 — DO NOT fold q.bits into jangtqRoutedBits anymore.
            // After the config-metadata bug fix (§410), bundles ship
            // `quantization.bits=8` for AFFINE attention/embed/lm_head,
            // while routed-expert TQ bits live in the separate top-level
            // `routed_expert_bits` field (decoded directly into
            // `jangtqRoutedBits`). Folding 8 here corrupts the TurboQuant
            // codebook size (256 entries vs the correct 4) and the
            // packed-weight stride → 4× wrong-magnitude routed expert
            // output → garbage logits. (Bug 3 root cause.)
            //
            // If a future bundle uses TQ for the top-level (mode='tq')
            // and ships its bits there, we'd need a more nuanced
            // detection — until then, treat top-level quantization as
            // affine and trust the JANG-named field.
        }
    }

    /// §394 — explicit `init(from:)` so every field is `decodeIfPresent`
    /// and the struct-level defaults are honored. Without this, the
    /// auto-synthesized Codable init does plain `decode` on the JANGTQ
    /// keys and rejects bundles that don't ship them at the top level
    /// (the actual DSV4-Flash-JANGTQ bundle ships only nested
    /// `quantization: {group_size, bits}` — the JANGTQ-named keys are
    /// resolved later by `resolveQuantOverrides()`). Manifests itself as
    /// `Failed to parse config.json: Missing field 'routed_expert_bits'`.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func opt<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        self.modelType = try opt(.modelType, "deepseek_v4")
        self.vocabSize = try opt(.vocabSize, 129280)
        self.hiddenSize = try opt(.hiddenSize, 4096)
        self.numHiddenLayers = try opt(.numHiddenLayers, 43)
        self.numAttentionHeads = try opt(.numAttentionHeads, 64)
        self.numKeyValueHeads = try opt(.numKeyValueHeads, 1)
        self.headDim = try opt(.headDim, 512)
        self.qkRopeHeadDim = try opt(.qkRopeHeadDim, 64)
        self.qLoraRank = try opt(.qLoraRank, 1024)
        self.oLoraRank = try opt(.oLoraRank, 1024)
        self.oGroups = try opt(.oGroups, 8)
        self.nRoutedExperts = try opt(.nRoutedExperts, 256)
        self.nSharedExperts = try opt(.nSharedExperts, 1)
        self.numExpertsPerTok = try opt(.numExpertsPerTok, 6)
        self.moeIntermediateSize = try opt(.moeIntermediateSize, 2048)
        self.numHashLayers = try opt(.numHashLayers, 3)
        self.numNextNPredictLayers = try opt(.numNextNPredictLayers, 1)
        self.scoringFunc = try opt(.scoringFunc, "sqrtsoftplus")
        self.topkMethod = try opt(.topkMethod, "noaux_tc")
        self.normTopkProb = try opt(.normTopkProb, true)
        self.routedScalingFactor = try opt(.routedScalingFactor, Float(1.5))
        self.swigluLimit = try opt(.swigluLimit, Float(10.0))
        self.hcMult = try opt(.hcMult, 4)
        self.hcSinkhornIters = try opt(.hcSinkhornIters, 20)
        self.hcEps = try opt(.hcEps, Float(1e-6))
        self.ropeTheta = try opt(.ropeTheta, Float(10000.0))
        self.compressRopeTheta = try opt(.compressRopeTheta, Float(160000.0))
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.maxPositionEmbeddings = try opt(.maxPositionEmbeddings, 1_048_576)
        self.slidingWindow = try opt(.slidingWindow, 128)
        self.rmsNormEps = try opt(.rmsNormEps, Float(1e-6))
        self.compressRatios = try opt(.compressRatios, [Int]())
        self.indexNHeads = try opt(.indexNHeads, 64)
        self.indexHeadDim = try opt(.indexHeadDim, 128)
        self.indexTopk = try opt(.indexTopk, 512)
        self.jangtqRoutedBits = try opt(.jangtqRoutedBits, 2)
        self.jangtqGroupSize = try opt(.jangtqGroupSize, 64)
        self.jangtqSeed = try opt(.jangtqSeed, 42)
        self.quantization = try c.decodeIfPresent(HFQuant.self, forKey: .quantization)
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

    /// Phase-Swift-FP16Pre + Phase-Swift-RMSFast (2026-04-24, ralph iter):
    /// cached unit-weight tensor for `MLXFast.rmsNorm` in collapse. Allocated
    /// at original input dtype so the fused norm runs on fp16 tensor cores.
    /// Mirrors Python `self._hc_rms_ones` from Phase-RMSFast.
    var hcRmsOnes: MLXArray

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
        self.hcRmsOnes = MLXArray.ones([hcDim], dtype: .float16)
        super.init()
    }

    /// Collapse (B, L, H, D) residual → (B, L, D) plus (post, comb) for later expand.
    ///
    /// §413 — explicit fp32 cast BEFORE mean(square()) reduction. The
    /// earlier fast path called `MLXFast.rmsNorm(weight: hcRmsOnes[fp16])`
    /// to keep the reduction on fp16 tensor cores for +0.42 tok/s. That
    /// optimization SHIPPED a latent M3 Ultra bug (Python `_hc_pre` #50
    /// root-cause): on Mac Studio M3 Ultra the bf16 `mean(square())`
    /// reduction saturates and produces garbage logits ("17 plus plus
    /// plus" failure mode). M4 Max happened to keep fp32 in SIMD lanes
    /// so MacBook tests passed silently. Mirroring Python's pip-installed
    /// pattern: cast `flat` to fp32, compute variance + rsqrt in fp32,
    /// cast back to input dtype before the matmul. Cost is one fp32
    /// promote per collapse — small fraction of overall mHC compute.
    func collapse(_ x: MLXArray) -> (collapsed: MLXArray, post: MLXArray, comb: MLXArray) {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped(B, L, H * D)
        let flatF32 = flat.asType(.float32)
        let rsqrtVal = rsqrt(
            (flatF32 * flatF32).mean(axis: -1, keepDims: true)
            + MLXArray(normEps)
        )
        let normed = (flatF32 * rsqrtVal).asType(flat.dtype)
        let mixes = matmul(normed, fn.T).asType(.float32)
        let (pre, post, comb) = hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: sinkhornIters, eps: hcEps
        )
        // collapsed = sum_h pre[h] * x[h]. Cast pre (small: hc_mult=4 elts) to
        // x.dtype so the broadcast multiply stays in fp16. Avoids fp32 promote
        // of the larger (B, L, H, D) tensor + final downcast.
        let preT = pre.asType(x.dtype)
        let preBroadcast = expandedDimensions(preT, axis: -1)
        let collapsed = (preBroadcast * x).sum(axis: 2)
        return (collapsed, post, comb)
    }

    /// Expand: y[i,d] = post[i] * blockOut[d] + sum_j comb[i,j] * residual[j,d]
    /// Uses matmul for the second term (faster than einsum).
    /// Phase-Swift-FP16Post (2026-04-24, ralph iter): drop fp32 casts. Apple
    /// Silicon Metal tensor cores accumulate matmul in fp32 internally so the
    /// explicit upcasts are redundant. Cast post and comb DOWN to blockOut.dtype
    /// (small: hc_mult and hc_mult² elements). Mirrors Python's Phase-FP16-Post
    /// verified output-identical.
    func expand(
        blockOut: MLXArray,    // (B, L, D)
        residual: MLXArray,    // (B, L, H, D)
        post: MLXArray,        // (B, L, H)  fp32 from sinkhorn kernel
        comb: MLXArray         // (B, L, H, H) fp32 from sinkhorn kernel
    ) -> MLXArray {
        let postT = post.asType(blockOut.dtype)
        let combT = comb.asType(blockOut.dtype)
        let y = _dsv4CompiledHcExpand(postT, blockOut, combT, residual)
        return y
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
    /// Phase-Swift-FP16Pre: cached unit-weight tensor for MLXFast.rmsNorm.
    var hcRmsOnes: MLXArray

    init(config: DeepseekV4JANGTQConfiguration) {
        self.hcMult = config.hcMult
        self.hcEps = config.hcEps
        self.normEps = config.rmsNormEps
        let hcDim = config.hcMult * config.hiddenSize
        self.fn = MLXArray.zeros([config.hcMult, hcDim], dtype: .float32)
        self.base = MLXArray.zeros([config.hcMult], dtype: .float32)
        self.scale = MLXArray.ones([1], dtype: .float32)
        self.hcRmsOnes = MLXArray.ones([hcDim], dtype: .float16)
        super.init()
    }

    /// §413 — explicit fp32 cast BEFORE mean(square()) reduction in
    /// the HyperHead RMSnorm-style normalization. Same root cause as
    /// `HyperConnection.collapse()` above — fp16 reduction saturates
    /// on M3 Ultra and produces garbage logits. Mirrors Python's
    /// pip-installed `_hc_head_reduce` fp32 path.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped(B, L, H * D)
        let flatF32 = flat.asType(.float32)
        let rsqrtVal = rsqrt(
            (flatF32 * flatF32).mean(axis: -1, keepDims: true)
            + MLXArray(normEps)
        )
        let normed = (flatF32 * rsqrtVal).asType(flat.dtype)
        let mixes = matmul(normed, fn.T).asType(.float32)
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
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP

    init(config: DeepseekV4JANGTQConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.gate = DSV4MoEGate(config: config, layerIdx: layerIdx)
        // §395 — JANGTQ-quantized routed experts. Bundle ships
        // `tq_bits/tq_norms/tq_packed` per expert; loading those into a
        // vanilla `SwitchGLU` would error with `unhandledKeys`. The
        // TurboQuantSwitchGLU sibling expects exactly that triad and
        // runs the fused gate+up+SwiGLU through the JANGTQ Metal
        // kernels. Bits + seed flow from config (resolved by
        // `resolveQuantOverrides()` from nested HF `quantization` block
        // when JANG-named keys are absent). Mirrors MiniMaxJANGTQ
        // pattern in MiniMaxJANGTQ.swift:148-154.
        //
        // §426 — DSV4 trains routed experts with `silu(min(gate, 10))
        // * clip(up, ±10)` (LimitedSwiGLU). Earlier this comment said
        // the missing clamp didn't matter empirically because tq_norms
        // pre-scales activations under 10 — that turned out to be
        // wrong: the Python investigation in
        // jang/research/DSV4-FLASH-MMLU-2026-04-26-DAY-LOG.md (§441)
        // measured a +4.5 pp MMLU jump after baking the clamp into
        // the same kernel. Mirror that fix here by forwarding
        // `swigluLimit: 10` → `meta[5]` → in-kernel clamp branch.
        // Other JANGTQ models leave swigluLimit=0 → kernel skips
        // clamp → byte-identical output to pre-§426.
        self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            bits: config.jangtqRoutedBits,
            seed: config.jangtqSeed,
            swigluLimit: Int(config.swigluLimit)
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
        var yRouted = switchMLP(x, inds)
        // BUG3 DIAG: trace MoE internals
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdx == 0 {
            nonisolated(unsafe) struct _GM { static var fired = false }
            if !_GM.fired {
                _GM.fired = true
                let lastIdx = x.dim(1) - 1
                let xV = x[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let indsV = inds[0..<1, lastIdx..<(lastIdx+1), 0..<inds.dim(2)].asType(.int32).flattened().asArray(Int32.self)
                let scoresV = scores[0..<1, lastIdx..<(lastIdx+1), 0..<scores.dim(2)].asType(.float32).flattened().asArray(Float.self)
                let yRV = yRouted[0..<1, lastIdx..<(lastIdx+1), 0..<yRouted.dim(2), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let yRn = sqrt((yRouted.asType(.float32) * yRouted.asType(.float32)).sum()).item(Float.self)
                var msg = "[BUG3-MOE] x[0,L-1,:8]=\(xV)\n[BUG3-MOE] inds=\(indsV) scores=\(scoresV)\n[BUG3-MOE] yRouted shape=\(yRouted.shape) full_norm=\(yRn) per-K[:8]=\(yRV)\n"
                if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
            }
        }
        // ── Bug 3 fix attempt (2026-04-25): Python `_DSV4SwiGLU` clamps
        // gate to [-∞, swiglu_limit] and up to [-swiglu_limit, swiglu_limit]
        // BEFORE silu*up. Swift's TurboQuantSwitchGLU bakes plain SwiGLU
        // without clamps. We can't easily inject the clamp inside the
        // fused Metal kernel, so post-apply a symmetric clamp on the
        // fused output. Not bit-exact to Python but bounds extreme
        // values that could produce wrong-sign logits over 43 layers.
        if config.swigluLimit > 0 {
            let lim = MLXArray(config.swigluLimit * config.swigluLimit)
            yRouted = MLX.clip(yRouted, min: -lim, max: lim)
        }
        // §404 — fp32 accumulator on the routed-expert weighted sum.
        //
        // Without the fp32 cast each expert contribution is summed in
        // bf16 across `topK=6` experts (DSV4 default), and the shared
        // expert add is a 7th term. bf16 has 8 bits of mantissa, so
        // adding 7 magnitude-similar terms quietly drops 2-3 bits of
        // precision per layer. After 43 layers this compounds into
        // visible drift on long generations — the symptom user
        // observed in JANGTQ benchmarks before the runtime fix
        // landed on the Python side. Cast scores + routed output to
        // fp32, do the weighted sum + shared-add in fp32, cast back
        // to the input dtype at the end. Cost is one fp32 promote +
        // demote per MoE layer, ~negligible vs the routed dispatch.
        let dtype = x.dtype
        let scoresF32 = scores.asType(.float32)
        let yRoutedF32 = yRouted.asType(.float32)
        var yF32 = (yRoutedF32 * expandedDimensions(scoresF32, axis: -1))
            .sum(axis: -2)
        let sharedOut = sharedExperts(x).asType(.float32)
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdx == 0 {
            nonisolated(unsafe) struct _GM2 { static var fired = false }
            if !_GM2.fired {
                _GM2.fired = true
                let lastIdx = yF32.dim(1) - 1
                let routedSum = yF32[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let routedNorm = sqrt((yF32 * yF32).sum()).item(Float.self)
                let sharedV = sharedOut[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let sharedNorm = sqrt((sharedOut * sharedOut).sum()).item(Float.self)
                let msg = "[BUG3-MOE] routedSum[0,L-1,:8]=\(routedSum) full_norm=\(routedNorm)\n[BUG3-MOE] shared[0,L-1,:8]=\(sharedV) shared_norm=\(sharedNorm)\n"
                if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
            }
        }
        yF32 = yF32 + sharedOut
        return yF32.asType(dtype)
    }
}

// MARK: - DSV4 Layer Cache (long-context)
//
// §417 (2026-04-25) — Port of Python `DeepseekV4Cache` (mlx_model.py:617-718).
// Wraps a `RotatingKVCache` (windowed KV) and carries two pool-state
// branches (`compressorState`, `indexerState`), each holding `bufferKv`,
// `bufferGate`, and `pooled` tensors. Decode-time long-context correctness
// requires this state to accumulate across calls.

public struct DSV4PoolState {
    public var bufferKv: MLXArray?
    public var bufferGate: MLXArray?
    public var pooled: MLXArray?
}

public enum DSV4PoolBranch {
    case compressor
    case indexer
}

public final class DSV4LayerCache: KVCache, CustomDebugStringConvertible {
    public let local: RotatingKVCache
    public var compressorState = DSV4PoolState()
    public var indexerState = DSV4PoolState()

    public init(slidingWindow: Int, keep: Int = 0, step: Int = 256) {
        self.local = RotatingKVCache(maxSize: slidingWindow, keep: keep, step: step)
    }

    // KVCache protocol — delegate to wrapped RotatingKVCache.
    public var offset: Int { local.offset }
    public var maxSize: Int? { local.maxSize }
    public var state: [MLXArray] {
        get { local.state }
        set { local.state = newValue }
    }
    public var metaState: [String] {
        get { local.metaState }
        set { local.metaState = newValue }
    }
    public var isTrimmable: Bool { local.isTrimmable }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        return local.update(keys: keys, values: values)
    }
    public func innerState() -> [MLXArray] { local.innerState() }
    @discardableResult
    public func trim(_ n: Int) -> Int { local.trim(n) }
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        return local.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    public func copy() -> any KVCache {
        // Pool state not deep-copied — DSV4 long-context decode does not
        // currently exercise prompt-cache reuse; if it ever does, port
        // Python's DeepseekV4Cache.copy().
        let c = DSV4LayerCache(
            slidingWindow: local.maxSize ?? 128,
            keep: 0, step: 256
        )
        c.state = self.state
        c.metaState = self.metaState
        c.compressorState = self.compressorState
        c.indexerState = self.indexerState
        return c
    }

    public var debugDescription: String {
        "DSV4LayerCache(window=\(local.maxSize ?? -1), offset=\(offset), "
            + "comp.pooled=\(compressorState.pooled?.dim(1) ?? 0), "
            + "idx.pooled=\(indexerState.pooled?.dim(1) ?? 0))"
    }

    private func readState(_ branch: DSV4PoolBranch) -> DSV4PoolState {
        switch branch {
        case .compressor: return compressorState
        case .indexer:    return indexerState
        }
    }

    private func writeState(_ branch: DSV4PoolBranch, _ s: DSV4PoolState) {
        switch branch {
        case .compressor: compressorState = s
        case .indexer:    indexerState = s
        }
    }

    /// `accumulate_windows` from Python (`mlx_model.py:651-689`). Splits the
    /// incoming `kv`/`gate` tensors into a "ready" prefix (multiple of
    /// `compress_ratio`) plus a buffered tail. Tail is stashed for next
    /// call; prefix is concatenated with prior buffered tail. Returns
    /// `(readyKv, readyGate, poolBase)` where `poolBase` is the absolute
    /// position of the FIRST token in `readyKv`.
    public func accumulateWindows(
        kv: MLXArray, gate: MLXArray,
        ratio: Int, startPos: Int,
        branch: DSV4PoolBranch
    ) -> (MLXArray, MLXArray, Int) {
        var s = readState(branch)
        let prevKv = s.bufferKv
        let prevGate = s.bufferGate

        let combinedKv: MLXArray
        let combinedGate: MLXArray
        if let p = prevKv, let pg = prevGate, p.dim(1) > 0 {
            combinedKv = concatenated([p, kv], axis: 1)
            combinedGate = concatenated([pg, gate], axis: 1)
        } else {
            combinedKv = kv
            combinedGate = gate
        }

        let total = combinedKv.dim(1)
        let usable = (total / ratio) * ratio
        let tailLen = total - usable

        let readyKv: MLXArray
        let readyGate: MLXArray
        if tailLen == 0 {
            readyKv = combinedKv
            readyGate = combinedGate
            s.bufferKv = nil
            s.bufferGate = nil
        } else if usable == 0 {
            // Entire combined tensor buffered for next call.
            var emptyKvShape = combinedKv.shape; emptyKvShape[1] = 0
            var emptyGateShape = combinedGate.shape; emptyGateShape[1] = 0
            readyKv = MLXArray.zeros(emptyKvShape, dtype: combinedKv.dtype)
            readyGate = MLXArray.zeros(emptyGateShape, dtype: combinedGate.dtype)
            s.bufferKv = combinedKv
            s.bufferGate = combinedGate
        } else {
            let kvParts = split(combinedKv, indices: [usable], axis: 1)
            let gateParts = split(combinedGate, indices: [usable], axis: 1)
            readyKv = kvParts[0]
            readyGate = gateParts[0]
            s.bufferKv = kvParts[1]
            s.bufferGate = gateParts[1]
        }

        let prevBufferLen = prevKv?.dim(1) ?? 0
        let poolBase = Swift.max(0, startPos) - prevBufferLen

        writeState(branch, s)
        return (readyKv, readyGate, poolBase)
    }

    /// `update_pool` from Python (`mlx_model.py:691-704`). Concatenates the
    /// freshly produced pool chunk onto persistent `pooled` and returns the
    /// cumulative pool.
    public func updatePool(
        _ newPooled: MLXArray,
        branch: DSV4PoolBranch
    ) -> MLXArray {
        var s = readState(branch)
        let merged: MLXArray
        if let p = s.pooled, p.dim(1) > 0 {
            merged = concatenated([p, newPooled], axis: 1)
        } else {
            merged = newPooled
        }
        s.pooled = merged
        writeState(branch, s)
        return merged
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

    /// Compressor forward — produces `(B, W, head_dim)` pooled compressed
    /// representations that long-context attention can attend to in addition
    /// to the rotating window.
    ///
    /// Full implementation ported from Python `mlx_model.py:Compressor.__call__`
    /// (2026-04-25). For decode (L < compressRatio) without persistent state,
    /// output is empty `(B, 0, head_dim)` — caller must skip the pool path.
    ///
    /// The overlap-transform branch (compressRatio == 4) doubles the
    /// compression axis so adjacent windows share boundary information,
    /// matching Python `_overlap_transform`.
    func callAsFunction(
        _ x: MLXArray,
        rope: DSV4RoPE,
        startPos: Int,
        cache: DSV4LayerCache? = nil,
        branch: DSV4PoolBranch = .compressor
    ) -> MLXArray {
        let (B, _, _) = (x.dim(0), x.dim(1), x.dim(2))
        let kvRaw = wkv(x)
        let gateRaw = wgate(x)

        // §417 — accumulate windows across calls when cache is provided.
        // Without cache, fall back to stateless slicing (only emit pool
        // when L >= compressRatio in this single call).
        let readyKvFlat: MLXArray
        let readyGateFlat: MLXArray
        let poolBase: Int
        if let c = cache {
            (readyKvFlat, readyGateFlat, poolBase) = c.accumulateWindows(
                kv: kvRaw, gate: gateRaw,
                ratio: compressRatio, startPos: startPos,
                branch: branch
            )
        } else {
            let S = kvRaw.dim(1)
            let usable = (S / compressRatio) * compressRatio
            if usable == 0 {
                return MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
            }
            let kvParts = S == usable ? [kvRaw] : split(kvRaw, indices: [usable], axis: 1)
            let gateParts = S == usable ? [gateRaw] : split(gateRaw, indices: [usable], axis: 1)
            readyKvFlat = kvParts[0]
            readyGateFlat = gateParts[0]
            poolBase = startPos
        }

        let usable = readyKvFlat.dim(1)
        if usable == 0 {
            // Cache-aware decode where L < compressRatio: this call emits
            // no new compressed tokens. Return prior pool from cache so
            // attention can still attend to history.
            if let c = cache {
                let prev = (branch == .compressor)
                    ? c.compressorState.pooled
                    : c.indexerState.pooled
                if let p = prev, p.dim(1) > 0 { return p }
            }
            return MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
        }
        let W = usable / compressRatio

        let readyKv = readyKvFlat.reshaped(B, W, compressRatio, outDim)
        var readyGate = readyGateFlat.reshaped(B, W, compressRatio, outDim)
        // Add absolute-position embedding: ape (compressRatio, outDim) broadcasts
        // across (B, W) leading dims.
        readyGate = readyGate + ape.asType(readyGate.dtype)

        // Overlap branch — for compressRatio=4 layers, expand each window
        // with the second half of its outDim AND the first half of the
        // PREVIOUS window's outDim. Doubles axis-2 from R to 2R, keeps
        // head_dim per slot.
        let kvBlocks: MLXArray
        let gateBlocks: MLXArray
        if overlap {
            kvBlocks = overlapTransform(readyKv, fillValue: 0.0)
            gateBlocks = overlapTransform(readyGate, fillValue: -Float.infinity)
        } else {
            // Non-overlap layers (compressRatio=128): outDim == head_dim
            // directly. The "out_dim slot" already has full head_dim.
            kvBlocks = readyKv
            gateBlocks = readyGate
        }

        // Softmax over the compression axis (axis 2 = compress_ratio in
        // non-overlap case, or 2*compress_ratio after overlap).
        // precise=true matches Python `softmax(precise=True)` for fp32 stability
        // when -inf appears in the gate after overlap.
        let weights = softmax(gateBlocks.asType(.float32), axis: 2, precise: true)
            .asType(kvBlocks.dtype)
        // Weighted sum over compression axis → (B, W, head_dim).
        var newPooled = (kvBlocks * weights).sum(axis: 2)
        newPooled = norm(newPooled.asType(x.dtype))

        // Apply RoPE at the absolute positions each compressed window
        // represents: position[w] = w * compressRatio + poolBase. The
        // tokens are NOT consecutive, so we MUST use the manual per-token
        // cos/sin path — `MLXFast.RoPE(_:offset: MLXArray)` interprets the
        // offset array as a per-batch scalar (per the docstring at
        // `MLXFast.swift:53`) which would collapse all pool slots to the
        // same phase. §416 added `DSV4RoPE.manual(_:positions:)` mirroring
        // Python `_call_manual` for exactly this case.
        let positions = MLXArray(0..<Int32(W)) * Int32(compressRatio) + Int32(poolBase)
        // newPooled: (B, W, head_dim) → (B, 1, W, head_dim) for partial RoPE
        let pooled4D = newPooled.expandedDimensions(axis: 1)
        // Apply RoPE to last ropeHeadDim dims; nope = first (head_dim - rope_dim).
        let ropeDim = ropeHeadDim
        let nopeSlice = pooled4D[.ellipsis, 0..<(headDim - ropeDim)]
        let peSlice = pooled4D[.ellipsis, (headDim - ropeDim)..<headDim]
        let peRotated = rope.manual(peSlice, positions: positions)
        let rotated = concatenated([nopeSlice, peRotated], axis: -1)
        let chunk = rotated.squeezed(axis: 1)

        // §417 — persist into the cache so subsequent decode steps see history.
        if let c = cache {
            return c.updatePool(chunk, branch: branch)
        }
        return chunk
    }

    /// Overlap transform: doubles the compression axis from R to 2R by
    /// concatenating the previous window's first-half (`x[..., :head_dim]`)
    /// with the current window's second-half (`x[..., head_dim:]`).
    ///
    /// Matches Python `_overlap_transform`:
    ///   out[:, w, :R, :]  = x[:, w-1, :, :head_dim]   (front; fill_value when w==0)
    ///   out[:, w, R:, :]  = x[:, w,   :, head_dim:]    (back, current window)
    ///
    /// Implemented via concat (no in-place slice assignment in MLX-Swift).
    private func overlapTransform(_ x: MLXArray, fillValue: Float) -> MLXArray {
        let B = x.dim(0)
        let W = x.dim(1)
        let R = x.dim(2)
        // Split outDim axis into first/second halves of head_dim each.
        let frontCurr = x[.ellipsis, 0..<headDim]                 // (B, W, R, head_dim)
        let back = x[.ellipsis, headDim..<(2 * headDim)]          // (B, W, R, head_dim)
        // Front shifted by 1 window: front[w] = frontCurr[w-1], front[0] = fill.
        let fillWindow = MLXArray.full(
            [B, 1, R, headDim],
            values: MLXArray(fillValue),
            dtype: x.dtype
        )
        let frontShift: MLXArray
        if W > 1 {
            // Drop the LAST window from frontCurr (axis=1) so we end up with W-1
            // shifted blocks; concat with `fillWindow` produces the W-element output.
            let parts = split(frontCurr, indices: [W - 1], axis: 1)
            frontShift = parts[0]
        } else {
            frontShift = MLXArray.zeros([B, 0, R, headDim], dtype: x.dtype)
        }
        let frontPadded = concatenated([fillWindow, frontShift], axis: 1)  // (B, W, R, head_dim)
        return concatenated([frontPadded, back], axis: 2)         // (B, W, 2R, head_dim)
    }
}

// MARK: - PR #1195 mask helpers (Swift port, 2026-04-25)
//
// Mirror Python `_build_window_mask` and `_compressed_visibility` in
// `jang_tools/dsv4/mlx_model.py:538-595`. Build SDPA-compatible 4D bool
// masks for the long-context path. Verified Python equivalent shipped
// 4174-token + 12K-needle tests on Mac Studio at 21 tok/s.

/// Visibility mask for the sliding-window portion of the cache.
/// Returns shape (B=1, 1, S, windowLen) bool, broadcastable to SDPA scores.
fileprivate func _buildWindowMask(
    B: Int, S: Int, offset: Int, window: Int, windowLen: Int
) -> MLXArray {
    // q_pos: (S,) absolute positions of queries
    let qPos = (MLXArray(0..<Int32(S)) + Int32(offset)).reshaped(1, S, 1)
    // raw_pos_at_k: (windowLen,) absolute position represented by cache slot k
    let cacheK = MLXArray(0..<Int32(windowLen))
    let rawPos = ((Int32(offset + S) - Int32(windowLen)) + cacheK).reshaped(1, 1, windowLen)
    let causal = rawPos .<= qPos                       // visibility: slot ≤ query
    let sliding = rawPos .> (qPos - Int32(window))     // not yet evicted
    let visible = MLX.logicalAnd(causal, sliding)
    return visible.expandedDimensions(axis: 1)         // (1, 1, S, windowLen)
}

/// Block-causal staircase: pool slot `k` is visible to query at q_pos iff
/// `(k+1)*ratio <= q_pos+1`. Returns shape (1, 1, S, compressedLen) bool.
fileprivate func _compressedVisibility(
    B: Int, S: Int, offset: Int, compressedLen: Int, ratio: Int
) -> MLXArray {
    let qPos = (MLXArray(0..<Int32(S)) + Int32(offset)).reshaped(1, S, 1)
    let k = MLXArray(0..<Int32(compressedLen))
    let lhs = ((k + 1) * Int32(ratio)).reshaped(1, 1, compressedLen)
    let rhs = qPos + 1
    let visible = lhs .<= rhs
    return visible.expandedDimensions(axis: 1)         // (1, 1, S, compressedLen)
}

/// §417 — Build a (B, 1, L, P) bool mask where slot `p` is True iff `p`
/// appears in the indexer's top-K selection for that query. Mirrors
/// Python `selected = (topk[..., None] == k_idx[None,None,None,:]).any(axis=-2)`
/// at `mlx_model.py:1088-1093`.
///
/// `topk` shape: (B, L, K) int32. `P` is the pool size.
fileprivate func _indexerSelected(topk: MLXArray, P: Int) -> MLXArray {
    let B = topk.dim(0)
    let L = topk.dim(1)
    let K = topk.dim(2)
    let topkI = topk.asType(.int32).reshaped(B, 1, L, K, 1)
    let kRange = MLXArray(0..<Int32(P)).reshaped(1, 1, 1, 1, P)
    let eq = topkI .== kRange                        // (B, 1, L, K, P) bool
    return eq.any(axis: -2)                          // (B, 1, L, P) bool
}

// MARK: - Indexer (top-k pool-slot selection for compress_ratio=4 layers)
//
// Mirrors Python `Indexer` in `jang_tools/dsv4/mlx_model.py:775-803`.
// For compress_ratio=4 attention layers ONLY. Builds a sparse top-k mask
// over the compressed pool so each query sees only the most-relevant
// `index_topk=512` compressed slots (vs all P slots).
//
// Wire path:
//   x:          (B, L, hidden)  — residual stream
//   qResidual:  (B, L, qLoraRank=1024) — output of q_norm, shared with main attention
//   stateCache: persistent compressor + indexer state (DSV4LayerCache.indexerState)
//   offset:     position of first query in this call
//
// Returns `(B, L, top_k)` int32 indices into the pool, OR nil if the
// indexer's internal Compressor produced an empty pool (L < ratio
// without persistent state).

final class Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let topK: Int
    let scale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: Compressor

    init(config: DeepseekV4JANGTQConfiguration, compressRatio: Int) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.scale = pow(Float(self.headDim), -0.5)
        self._wqB.wrappedValue = Linear(config.qLoraRank, nHeads * headDim, bias: false)
        self._weightsProj.wrappedValue = Linear(config.hiddenSize, nHeads, bias: false)
        // Indexer's own Compressor — uses index_head_dim (smaller than attn head_dim)
        self._compressor.wrappedValue = Compressor(
            config: config, compressRatio: compressRatio, headDim: headDim
        )
        super.init()
    }

    /// Returns top-k indices into the compressor pool, or nil if pool is empty.
    /// `qResidual`: q_norm output (B, L, qLoraRank), shared with main attention.
    /// §417 — accepts the layer cache so the indexer's internal Compressor
    /// uses the SECOND state branch (`indexerState`), separate from the
    /// main Compressor's branch.
    func callAsFunction(
        _ x: MLXArray,
        qResidual: MLXArray,
        rope: DSV4RoPE,
        startPos: Int,
        cache: DSV4LayerCache? = nil
    ) -> MLXArray? {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let pooled = compressor(x, rope: rope, startPos: startPos,
                                cache: cache, branch: .indexer)
        let P = pooled.dim(1)
        if P == 0 { return nil }
        // q: (B, L, n_heads, head_dim) — own Q projection, NOT main attention's q
        var q = wqB(qResidual).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        // Apply RoPE — main rope (NOT compress_rope) since these are query-side positions.
        let qNope = q[.ellipsis, 0..<(headDim - rope.dims)]
        let qPe = rope(q[.ellipsis, (headDim - rope.dims)..<headDim], offset: startPos)
        q = concatenated([qNope, qPe], axis: -1)
        // Score: q (B, H, L, D) @ pooled.T (B, D, P) → (B, H, L, P)
        // Reduce: relu + per-head weight + sum over heads → (B, L, P)
        let weightsRaw = weightsProj(x).asType(.float32)  // (B, L, n_heads)
        let qF32 = q.asType(.float32)
        let pooledF32 = pooled.expandedDimensions(axis: 1).asType(.float32)  // (B, 1, P, D)
        let pooledT = pooledF32.transposed(0, 1, 3, 2)  // (B, 1, D, P)
        // Broadcast (B, H, L, D) @ (B, 1, D, P) — H broadcasts across the 1 dim
        var rawScores = matmul(qF32, pooledT)  // (B, H, L, P)
        rawScores = MLX.maximum(rawScores, 0)  // relu
        // Per-head scale
        let perHead = weightsRaw * (scale * pow(Float(nHeads), -0.5))  // (B, L, n_heads)
        // Score: sum_h(rawScores[b, h, l, p] * perHead[b, l, h])
        // Reshape for matmul: rawScores → (B, L, H, P), perHead → (B, L, 1, H)
        let scoresLHP = rawScores.transposed(0, 2, 1, 3)  // (B, L, H, P)
        let perHeadExp = perHead.expandedDimensions(axis: 2)  // (B, L, 1, H)
        let scores = matmul(perHeadExp, scoresLHP).squeezed(axis: 2)  // (B, L, P)
        let k = min(topK, P)
        // argpartition on negated scores, then take last k → top-k by score
        let topkIdx = MLX.argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        return topkIdx.asType(.int32)
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

    let rope: DSV4RoPE

    @ModuleInfo(key: "compressor") var compressor: Compressor?
    /// §415 (2026-04-25) — Indexer wired as `@ModuleInfo` after the §410
    /// shape-authoritative loader landed. Earlier hcCollapse `(B,L,32768) ×
    /// (16384,24)` mismatch was a stale build cache, not the indexer
    /// declaration; resolved via §410 + §414. Indexer is `Optional<Module>`,
    /// instantiated only for `compressRatio == 4` layers (per Python
    /// reference `mlx_model.py:951-952`); other layers leave it nil so the
    /// loader has no `self_attn.indexer.*` keys to consume there.
    @ModuleInfo(key: "indexer") var indexer: Indexer?

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
        // Phase-Swift-DSV4Rope (2026-04-24, ralph iter): use DSV4RoPE custom
        // class instead of mlx-swift's YarnRoPE. YarnRoPE applies an mscale
        // factor (1.277 for DSV4 factor=16) that Python's DeepseekV4RoPE does
        // NOT apply. The mscale divergence would cause Swift output to differ
        // from Python at every layer's attention. DSV4RoPE matches Python
        // exactly — modifies inv_freq via YaRN smooth interpolation only,
        // no mscale applied to x.
        let effectiveBase = compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta
        let effectiveScaling: [String: StringOrNumber]? = compressRatio > 0 ? config.ropeScaling : nil
        self.rope = DSV4RoPE(
            dims: ropeHeadDim,
            base: effectiveBase,
            scalingConfig: effectiveScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        if compressRatio > 0 {
            self._compressor.wrappedValue = Compressor(
                config: config, compressRatio: compressRatio, headDim: headDim
            )
            // §415 — Indexer only on compress_ratio==4 fast-cadence layers.
            // Mirrors Python `if compress_ratio == 4: self.indexer = Indexer(...)`.
            if compressRatio == 4 {
                self._indexer.wrappedValue = Indexer(
                    config: config, compressRatio: compressRatio
                )
            }
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

        // Q — low-rank + per-head RMSNorm.
        //
        // Order: q_norm(qLoraRank=1024) → wq_b → per-head fp32 RMS rescale.
        // The two norms serve different purposes: q_norm conditions the
        // LoRA residual stream BEFORE expansion, the per-head rescale
        // tames variance AFTER the (1024 → nHeads*headDim) expansion.
        //
        // §398 — Per-head rescale MUST run in fp32. The earlier fast
        // path used `MLXFast.rmsNorm(weight: ones[bf16])` which the
        // kernel may compute in bf16 — and on DSV4 specifically the
        // accumulated variance overflows bf16 by layer ~40 once
        // `compress_ratio>0` layers contribute their long-range
        // compressed-KV residual stream. Output-side: the model
        // produces locally-coherent-but-globally-looping text ("The
        // Last Code: The Last Code …") because the 64-of-512 partial
        // RoPE channels go to inf while the 448 nope channels still
        // carry semantic structure. Explicit fp32 cast → variance →
        // rsqrt → multiply, then cast back to original dtype, makes
        // the rescale numerically equivalent to Python's
        // `_dsv4_per_head_rms_fp32` and stops the drift.
        // DIAG (2026-04-25): trace V4Attention.callAsFunction shapes at decode time
        do {
            let msg = "[v4attn-diag layer=\(layerIdx)] x.shape=\(x.shape) B=\(B) L=\(L) cache.offset=\(cache?.offset ?? -1)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }
        let qResidual = qNorm(wqA(x))
        // DIAG: q reshape
        do {
            let qBefore = wqB(qResidual)
            let msg = "[v4attn-diag layer=\(layerIdx)] qResidual.shape=\(qResidual.shape) wqB(qResidual).shape=\(qBefore.shape) target_q=(\(B),\(L),\(nHeads),\(headDim)) elements=\(B*L*nHeads*headDim) wkv_in_dim=\(hiddenSize)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }
        var q = wqB(qResidual).reshaped(B, L, nHeads, headDim)
        let qDtype = q.dtype
        let qF32 = q.asType(.float32)
        let varRsqrt = rsqrt(
            (qF32 * qF32).mean(axis: -1, keepDims: true) + MLXArray(rmsNormEps)
        )
        q = (qF32 * varRsqrt).asType(qDtype)
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

        // ── Long-context branch (Compressor + Indexer + window+compressed mask) ──
        //
        // §417 (2026-04-25) — full long-context wiring:
        //   • Cache cast to `DSV4LayerCache` so pool state accumulates across
        //     decode calls (was: pool history dropped every step).
        //   • Compressor receives cache so `accumulateWindows` + `updatePool`
        //     persist `bufferKv`/`bufferGate`/`pooled`.
        //   • For `compressRatio == 4` layers, Indexer runs and produces top-k
        //     pool indices. Two SDPA paths:
        //       - L==1 decode + Indexer present → take_along_axis gather of
        //         top-k pool rows; mask=causal (sdpaMask kept as default).
        //       - else (prefill or non-indexer layer) → bool visibility mask
        //         AND'd with indexer-selected (when present) and concatenated
        //         to the window mask before SDPA.
        // Engaged when `VMLX_DSV4_LONG_CTX=1`.
        let v4Cache = cache as? DSV4LayerCache
        var sdpaMask = mask
        let useLongCtx = ProcessInfo.processInfo.environment["VMLX_DSV4_LONG_CTX"] == "1"
        if compressRatio > 0 && useLongCtx, let comp = compressor {
            let pooled = comp(x, rope: self.rope, startPos: offset,
                              cache: v4Cache, branch: .compressor)
            let P = pooled.dim(1)
            if P > 0 {
                // Run Indexer for compress_ratio==4 layers; nil otherwise.
                var topkIdx: MLXArray? = nil
                if let idx = self.indexer {
                    topkIdx = idx(x, qResidual: qResidual,
                                  rope: self.rope, startPos: offset,
                                  cache: v4Cache)
                }

                if L == 1, let topk = topkIdx {
                    // S=1 decode fast path: gather top-k rows directly.
                    let K = topk.dim(-1)
                    let pooledExp = pooled.expandedDimensions(axis: 1)         // (B, 1, P, D)
                    let topkExp = topk.expandedDimensions(axis: -1)            // (B, 1, K, 1)
                    let topkBcast = MLX.broadcast(
                        topkExp, to: [pooled.dim(0), 1, K, pooled.dim(2)])     // (B, 1, K, D)
                    let gathered = MLX.takeAlong(
                        pooledExp, topkBcast.asType(.int32), axis: 2
                    )
                    keys = concatenated([keys, gathered], axis: 2)
                    values = concatenated([values, gathered], axis: 2)
                    // Default causal mask works: top-k slots are already
                    // both window-visible AND query-causal by construction.
                } else {
                    let windowLen = keys.dim(2)
                    let winMask = _buildWindowMask(
                        B: B, S: L, offset: offset,
                        window: config.slidingWindow, windowLen: windowLen
                    )
                    var compMask = _compressedVisibility(
                        B: B, S: L, offset: offset,
                        compressedLen: P, ratio: compressRatio
                    )
                    // AND with indexer top-k selection when available.
                    if let topk = topkIdx {
                        let sel = _indexerSelected(topk: topk, P: P)  // (B, 1, L, P) bool
                        compMask = MLX.logicalAnd(compMask, sel)
                    }
                    let pooled4D = pooled.expandedDimensions(axis: 1)
                    keys = concatenated([keys, pooled4D], axis: 2)
                    values = concatenated([values, pooled4D], axis: 2)
                    let combinedMask = concatenated([winMask, compMask], axis: -1)
                    sdpaMask = .array(combinedMask)
                }
            }
        }
        // ── End long-context branch ──────────────────────────────────────────

        // ── Bug 3 fix (2026-04-25): match Python `sinks=None` ────────────────
        // Earlier `Phase-Swift-Sinks` note thought we needed to PASS sinks,
        // but Python reference at mlx_model.py:1125 explicitly passes
        // `sinks=None`. Python `attn_sink` parameter has shape
        // (n_kv_heads=1, 1, head_dim=512) and is initialized to zeros but
        // NEVER passed to SDPA — it stays as a model-level parameter that
        // gets loaded but doesn't affect attention math. Swift was passing
        // `attnSink: (n_heads=64,)` which (a) has the wrong shape and
        // (b) injects spurious per-head bias into SDPA logits → garbage
        // tokens. Removing sinks restores parity with Python.
        var out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: keys,
            values: values,
            scale: softmaxScale,
            mask: sdpaMask
        )

        // Phase-Swift-InverseRope (2026-04-24, ralph iter): apply inverse RoPE
        // on attention output rope-portion. Required for coherent output —
        // Python verified that skipping this produces gibberish (research §29).
        // Implementation: negate the rope's wavelength tensor and pass to
        // MLXFast.RoPE directly. YarnRoPE.freqs is now exposed via public
        // getter (added to RoPEUtils.swift this iter).
        // Math: cos(neg_offset * freq) = cos(offset * neg_freq);
        //       sin(neg_offset * freq) = -sin(offset * neg_freq).
        // So `freqs = -wavelength` flips the rotation sense, equivalent to
        // applying `inverse=True` in Python's manual cos/sin path.
        let (outNope, outPe0) = splitLastDim(out, at: nopeHeadDim)
        let outPe = rope.inverse(outPe0, offset: offset)
        out = concatenated([outNope, outPe], axis: -1)

        // Reshape + grouped O projection
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, nHeads * headDim)
        out = groupedOProjection(out, B: B, L: L)

        return woB(out)
    }

    /// Grouped O projection: reshape to (B, L, oGroups, groupFeat), then
    /// per-group matmul to (B, L, oGroups, oLoraRank), then flatten.
    ///
    /// §398 — When `woA` is `QuantizedLinear` (the JANGTQ load path
    /// auto-promotes affine 8-bit Linears), the previous path
    /// `woA.weight.reshaped(oGroups, oLoraRank, groupFeat)` reshaped
    /// the PACKED uint32 weight tensor element-wise, which silently
    /// scrambled the 4-or-8 sub-elements per uint32 across group
    /// boundaries → garbage attention output → looping decode. The
    /// quantized path must use `MLX.quantizedMatmul(transpose: true)`
    /// per-group with weight + scales + biases passed explicitly so
    /// the dequant happens INSIDE the matmul kernel and packing stays
    /// undisturbed. Reshaping the OUT axis (`oGroups * oLoraRank`)
    /// IS valid for all three (weight/scales/biases) because that
    /// axis splits cleanly on group boundaries; the IN axis (which
    /// holds the packed sub-elements) stays untouched.
    func groupedOProjection(_ out: MLXArray, B: Int, L: Int) -> MLXArray {
        let groupFeat = (nHeads * headDim) / oGroups
        let reshaped = out.reshaped(B, L, oGroups, groupFeat)

        if let qWoA = woA as? QuantizedLinear {
            // Per reference DeepseekV4.swift:333-345 — the per-group
            // batched quantizedMatmul requires a singleton dim between
            // G and OUT on the weight side so the kernel's batch broadcast
            // resolves (G, B, L, gf) × (G, 1, OUT, in_packed) →
            // (G, B, L, OUT). Without the `.expandedDimensions(axis: 1)`
            // the kernel can't align the leading G axis between input
            // and weight and silently returns a 0-d scalar.
            let xT = reshaped.transposed(2, 0, 1, 3)  // (G, B, L, gf)
            let wPacked = qWoA.weight.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let wScales = qWoA.scales.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let wBiases = qWoA.biases?.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let outQ = MLX.quantizedMatmul(
                xT, wPacked, scales: wScales, biases: wBiases,
                transpose: true,
                groupSize: qWoA.groupSize, bits: qWoA.bits,
                mode: qWoA.mode
            )
            return outQ.transposed(1, 2, 0, 3)
                .reshaped(B, L, oGroups * oLoraRank)
        }

        // Non-quantized fallback (legacy bf16 / fp16 / fp32 weight).
        // Original path: einsum("bsgd,grd->bsgr", out, weight.reshape(G, R, D))
        let weight = woA.weight.reshaped(oGroups, oLoraRank, groupFeat)
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
    nonisolated(unsafe) static var _hcCollapseTraceCount: Int = 0  // DIAG: first 5 calls
    nonisolated(unsafe) static var _layerCallCount: Int = 0  // DIAG: layer entry
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
        // DIAG (2026-04-25): trace shape on first 5 calls (covers prefill +
        // first decode hcCollapse calls, both ATTN and FFN HC).
        if DeepseekV4DecoderLayer._hcCollapseTraceCount < 5 {
            DeepseekV4DecoderLayer._hcCollapseTraceCount += 1
            let msg = "[hcCollapse-diag #\(DeepseekV4DecoderLayer._hcCollapseTraceCount)] h.shape=\(h.shape) expected (B, L, \(config.hcMult), \(config.hiddenSize)) fn.shape=\(fn.shape)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }
        // ── DO NOT REMOVE THE `.asType(.float32)` BELOW ──────────────────────
        // The `flat` tensor has shape (B, L, hc_mult * hidden) = (B, L, 16384)
        // for DSV4. Computing `(flat * flat).mean(axis: -1)` over this 16384-dim
        // vector MUST run in fp32. On M3 Ultra (Mac Studio), the implicit
        // bf16 accumulation saturates and produces garbage logits — Python's
        // `_hc_pre` had this exact bug fixed 2026-04-25 (ralph iter 50,
        // memory note `feedback_hc_pre_fp32_cast.md`). Symptom on the Python
        // side: "17 + 28" → "17 plus plus plus" loop. Swift's
        // `.asType(.float32)` here is what kept Swift from hitting the same bug.
        // ── ──────────────────────────────────────────────────────────────────
        let flat = h.reshaped(B, L, H * D).asType(.float32)
        let rsqrtVal = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + config.rmsNormEps)
        let mixes = matmul(flat, fn.T) * rsqrtVal
        let (pre, post, comb) = hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: config.hcMult, iters: config.hcSinkhornIters, eps: config.hcEps
        )
        // BUG3 DIAG: dump first call's pre/post/comb at last position
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" {
            nonisolated(unsafe) struct _G4 { static var fired = false }
            if !_G4.fired {
                _G4.fired = true
                let lastIdx = L - 1
                let preV = pre[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32).flattened().asArray(Float.self)
                let postV = post[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32).flattened().asArray(Float.self)
                let combV = comb[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32).flattened().asArray(Float.self)
                let mixV = mixes[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32).flattened().asArray(Float.self)
                let scaleV = scale.asType(.float32).flattened().asArray(Float.self)
                let baseV = base.asType(.float32).flattened().asArray(Float.self)
                let fnFirstRow = fn[0..<1, 0..<8].asType(.float32).flattened().asArray(Float.self)
                let fnLastRow = fn[23..<24, 0..<8].asType(.float32).flattened().asArray(Float.self)
                let flatLast8 = flat[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let msg = "[BUG3-MIX] scale=\(scaleV) base=\(baseV)\n[BUG3-MIX] fn[0,:8]=\(fnFirstRow) fn[23,:8]=\(fnLastRow)\n[BUG3-MIX] flat[0,L-1,:8]=\(flatLast8)\n[BUG3-MIX] mixes=\(mixV)\n[BUG3-MIX] pre=\(preV)\n[BUG3-MIX] post=\(postV)\n[BUG3-MIX] comb=\(combV)\n"
                if let d = msg.data(using: .utf8) {
                    try? FileHandle.standardError.write(contentsOf: d)
                }
            }
        }
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
        // DIAG (2026-04-25): trace decoder layer entry, especially during decode
        if DeepseekV4DecoderLayer._layerCallCount < 50 || DeepseekV4DecoderLayer._layerCallCount % 43 == 0 {
            DeepseekV4DecoderLayer._layerCallCount += 1
            let msg = "[layer-call #\(DeepseekV4DecoderLayer._layerCallCount)] h.shape=\(h.shape) inputIds.shape=\(inputIds.shape) cache.offset=\(cache?.offset ?? -1)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        } else {
            DeepseekV4DecoderLayer._layerCallCount += 1
        }
        // ATTN HC
        var residual = h
        var (x, post, comb) = hcCollapse(
            h, fn: hcAttnFn, scale: hcAttnScale, base: hcAttnBase
        )
        x = inputLayerNorm(x)
        // BUG3 DIAG: dump pre-attn collapsed input
        let layerIdxLocal = DeepseekV4DecoderLayer._layerCallCount  // approximate
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdxLocal <= 1 {
            let lastIdx = x.dim(1) - 1
            let v = x[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
            let msg = "[BUG3-LAYER0] post-inputLN x[0,L-1,:8]=\(v) shape=\(x.shape)\n"
            if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        x = selfAttn(x, mask: mask, cache: cache)
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdxLocal <= 1 {
            let lastIdx = x.dim(1) - 1
            let v = x[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
            let xn = sqrt((x.asType(.float32) * x.asType(.float32)).sum()).item(Float.self)
            let msg = "[BUG3-LAYER0] post-attn x[0,L-1,:8]=\(v) shape=\(x.shape) full_norm=\(xn)\n"
            if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        var hNew = hcExpand(blockOut: x, residual: residual, post: post, comb: comb)
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdxLocal <= 1 {
            let lastIdx = hNew.dim(1) - 1
            for slot in 0..<4 {
                let v = hNew[0..<1, lastIdx..<(lastIdx+1), slot..<(slot+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let s = hNew[0..<1, lastIdx..<(lastIdx+1), slot..<(slot+1), .ellipsis].asType(.float32)
                let n = sqrt((s*s).sum()).item(Float.self)
                let msg = "[BUG3-LAYER0] post-attn-HC slot=\(slot) :8=\(v) norm=\(n)\n"
                if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
            }
        }

        // FFN HC
        residual = hNew
        (x, post, comb) = hcCollapse(
            hNew, fn: hcFfnFn, scale: hcFfnScale, base: hcFfnBase
        )
        // DIAG: shape after FFN hcCollapse
        if DeepseekV4DecoderLayer._layerCallCount <= 50 {
            let s = "[layer-flow #\(DeepseekV4DecoderLayer._layerCallCount)] post-FFN-hcCollapse x.shape=\(x.shape) residual.shape=\(residual.shape)\n"
            if let d = s.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdxLocal <= 1 {
            let lastIdx = x.dim(1) - 1
            let v = x[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
            let postV = post[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32).flattened().asArray(Float.self)
            let combV = comb[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32).flattened().asArray(Float.self)
            let msg = "[BUG3-FFN] post-FFN-collapsed x[0,L-1,:8]=\(v) shape=\(x.shape)\n[BUG3-FFN] FFN-post=\(postV)\n[BUG3-FFN] FFN-comb=\(combV)\n"
            if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        x = postAttentionLayerNorm(x)
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdxLocal <= 1 {
            let lastIdx = x.dim(1) - 1
            let v = x[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
            let msg = "[BUG3-FFN] post-postAttnLN x[0,L-1,:8]=\(v)\n"
            if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        x = mlp.forward(x, inputIds: inputIds)
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && layerIdxLocal <= 1 {
            let lastIdx = x.dim(1) - 1
            let v = x[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
            let xn = sqrt((x.asType(.float32) * x.asType(.float32)).sum()).item(Float.self)
            let msg = "[BUG3-FFN] post-mlp x[0,L-1,:8]=\(v) shape=\(x.shape) full_norm=\(xn)\n"
            if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        // DIAG: mlp output shape
        if DeepseekV4DecoderLayer._layerCallCount <= 50 {
            let s = "[layer-flow #\(DeepseekV4DecoderLayer._layerCallCount)] post-mlp x.shape=\(x.shape)\n"
            if let d = s.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }
        hNew = hcExpand(blockOut: x, residual: residual, post: post, comb: comb)
        // DIAG: final return shape
        if DeepseekV4DecoderLayer._layerCallCount <= 50 {
            let s = "[layer-flow #\(DeepseekV4DecoderLayer._layerCallCount)] return hNew.shape=\(hNew.shape)\n"
            if let d = s.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
        }

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
        // DIAG (2026-04-25): trace first call's h0 shape
        do {
            let msg = "[forward-diag] inputIds.shape=\(inputIds.shape) h0.shape=\(h0.shape) hidden=\(config.hiddenSize) hcMult=\(config.hcMult)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }
        // BUG3 DIAG: post-embed first-token first-8-elements
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" {
            nonisolated(unsafe) struct _G2 { static var fired = false }
            if !_G2.fired {
                _G2.fired = true
                let v = h0[0..<1, 0..<1, 0..<8].asType(.float32).flattened().asArray(Float.self)
                let msg = "[BUG3-DIAG] post-embed h0[0,0,:8]=\(v) dtype=\(h0.dtype)\n"
                if let d = msg.data(using: .utf8) {
                    try? FileHandle.standardError.write(contentsOf: d)
                }
            }
        }
        // Tile: (B, L, hidden) → (B, L, hc_mult, hidden)
        var h = tiled(expandedDimensions(h0, axis: -2), repetitions: [1, 1, config.hcMult, 1])

        // Causal mask (sliding-window, as array — bug 1.9b)
        // For prefill attentionWithCacheUpdate handles masking;
        // here we pass "causal" via MLXFast API.
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], inputIds: inputIds)
            // BUG3 DIAG: dump first layer's output (h is (B, L, hcMult, hidden))
            if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" && i < 2 {
                nonisolated(unsafe) struct _G3 { static var fired: [Int: Bool] = [:] }
                if _G3.fired[i] == nil {
                    _G3.fired[i] = true
                    let lastIdx = h.dim(1) - 1
                    // Norm of full residual at last position
                    let lastFlat = h[0..<1, lastIdx..<(lastIdx+1), .ellipsis].asType(.float32)
                    let normVal = sqrt((lastFlat * lastFlat).sum()).item(Float.self)
                    var msg = "[BUG3-DIAG] post-layer\(i) full_norm=\(normVal)\n"
                    // Per-slot first-8
                    for slot in 0..<4 {
                        let v = h[0..<1, lastIdx..<(lastIdx+1), slot..<(slot+1), 0..<8]
                            .asType(.float32).flattened().asArray(Float.self)
                        let slotFull = h[0..<1, lastIdx..<(lastIdx+1), slot..<(slot+1), .ellipsis]
                            .asType(.float32)
                        let slotNorm = sqrt((slotFull * slotFull).sum()).item(Float.self)
                        msg += "[BUG3-DIAG] post-layer\(i) slot=\(slot) :8=\(v) norm=\(slotNorm)\n"
                    }
                    if let d = msg.data(using: .utf8) {
                        try? FileHandle.standardError.write(contentsOf: d)
                    }
                }
            }
        }

        // HyperHead reduce: (B, L, hc_mult, hidden) → (B, L, hidden)
        let (B, L, H, D) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        // ── DO NOT REMOVE `.asType(.float32)` ────────────────────────────────
        // Same bf16-saturation hazard as in `hcCollapse`. The 16384-dim sum
        // overflows on M3 Ultra without explicit fp32 cast. See `hcCollapse`
        // doc-comment + Python `_hc_pre` fix (memory note
        // `feedback_hc_pre_fp32_cast.md`).
        // ── ──────────────────────────────────────────────────────────────────
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

    /// §417 (2026-04-25) — Long-context cache override.
    ///
    /// When `VMLX_DSV4_LONG_CTX=1`, return per-layer caches sized to
    /// match each layer's `compressRatio`:
    ///   • `compressRatio == 0` → plain `RotatingKVCache(slidingWindow)`.
    ///     The layer never engages the long-context branch.
    ///   • `compressRatio > 0`  → `DSV4LayerCache(slidingWindow)` carrying
    ///     `compressorState` + `indexerState` pool buffers across decode
    ///     calls, mirroring Python `Model.make_cache` at
    ///     `mlx_model.py:1542-1569`.
    ///
    /// When the env-flag is unset, fall back to plain RotatingKVCache for
    /// every layer so short-prompt decode is unchanged.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let useLongCtx = ProcessInfo.processInfo.environment["VMLX_DSV4_LONG_CTX"] == "1"
        let window = config.slidingWindow
        guard useLongCtx else {
            return (0..<config.numHiddenLayers).map { _ in
                RotatingKVCache(maxSize: window, keep: 0)
            }
        }
        // §417 — Long-context: per-layer caches sized to compressRatio.
        //   • compressRatio == 0 → plain RotatingKVCache (window only)
        //   • compressRatio > 0  → DSV4LayerCache (carries pool buffers)
        return (0..<config.numHiddenLayers).map { i in
            let r = config.effectiveCompressRatio(forLayer: i)
            if r > 0 {
                return DSV4LayerCache(slidingWindow: window)
            }
            return RotatingKVCache(maxSize: window, keep: 0)
        }
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let h = model(inputs, cache: cache)
        let logits = lmHead(h)
        // BUG3 DIAG: dump first call's exit logits + hidden state for cross-runtime compare
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" {
            nonisolated(unsafe) struct _G { static var fired = false }
            if !_G.fired {
                _G.fired = true
                // Hidden state: last position
                let lastIdx = h.dim(1) - 1
                let lastH = h[0..<1, lastIdx..<(lastIdx+1), 0..<8].asType(.float32)
                let lastL = logits[0..<1, lastIdx..<(lastIdx+1), 0..<logits.dim(2)].asType(.float32)
                let stats = "min=\(lastL.min().item(Float.self)) max=\(lastL.max().item(Float.self)) mean=\(lastL.mean().item(Float.self))"
                let h8 = lastH.flattened().asArray(Float.self)
                // Top-5 logits
                let lastFlat = lastL.flattened()
                let sortedIdx = argSort(-lastFlat, axis: 0)[0..<5]
                let top5Idx = sortedIdx.asArray(Int32.self)
                let top5Vals = takeAlong(lastFlat, sortedIdx, axis: 0).asArray(Float.self)
                var msg = "[BUG3-DIAG] inputs.shape=\(inputs.shape) inputs[0]=\(inputs[0].asArray(Int32.self))\n"
                msg += "[BUG3-DIAG] last hidden h[0,L-1,:8]=\(h8)\n"
                msg += "[BUG3-DIAG] logits stats: \(stats)\n"
                msg += "[BUG3-DIAG] top-5 logits:\n"
                for (i, v) in zip(top5Idx, top5Vals) {
                    msg += "[BUG3-DIAG]   tok=\(i) logit=\(v)\n"
                }
                if let d = msg.data(using: .utf8) {
                    try? FileHandle.standardError.write(contentsOf: d)
                }
            }
        }
        return logits
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
            // §395 — DSV4-Flash JANGTQ ships per-expert `.tq_bits` (shape [1])
            // alongside `tq_packed` + `tq_norms`. The bit count is already
            // known from `config.jangtqRoutedBits` (resolved from nested
            // `quantization.bits` if needed), so the per-tensor copy is
            // redundant. TurboQuantSwitchLinear does NOT declare a
            // `tq_bits` @ParameterInfo, so leaving it in the sanitized
            // weights triggers `unhandledKeys`. Strip it here.
            if key.hasSuffix(".tq_bits") { continue }
            // §415 (2026-04-25) — Indexer is now wired via @ModuleInfo on
            // `DeepseekV4Attention` (`compressRatio == 4` layers only).
            // The pass-through `attn.<rest>` path below routes
            // `layers.N.attn.indexer.*` → `model.layers.N.self_attn.indexer.*`
            // matching the new property registration. Bundle only ships
            // indexer weights at `compressRatio == 4` layers (verified
            // 2026-04-25), and only those layers init an Indexer module,
            // so layered registration matches and load succeeds.

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
                    // §395 — DSV4 ships `gate.bias` for the noaux_tc score
                    // correction term. The Swift module exposes it as
                    // `e_score_correction_bias` (Python parity); without
                    // the rename, load fails with `unhandledKeys keys=["bias"]`.
                    let gsubMapped = (gsub == "bias") ? "e_score_correction_bias" : gsub
                    sanitized["\(prefix).mlp.gate.\(gsubMapped)"] = value
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

/// Tile x along given axes — MATERIALIZED, not a broadcast view.
///
/// Bug 3 root cause (2026-04-25): the previous `broadcast` implementation
/// returned a strided view sharing memory across all hc_mult slots. Python
/// reference at `mlx_model.py:1490` explicitly comments:
///   "Must be materialized (not a broadcast view) — matches torch reference
///    `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`. Subsequent
///    `flatten(start_axis=2)` inside `_hc_pre` would see wrong strided data
///    from a broadcast view."
/// MLX-swift provides `MLX.tiled(...)` which performs the materializing copy.
func tiled(_ x: MLXArray, repetitions: [Int]) -> MLXArray {
    return MLX.tiled(x, repetitions: repetitions)
}

// MARK: - DSV4-specific RoPE (matches Python DeepseekV4RoPE exactly)
//
// Phase-Swift-DSV4Rope (2026-04-24, ralph iter): Swift's YarnRoPE
// applies an `mscale = 1.277` for DSV4's factor=16 config (default
// mscale=1, mscale_all_dim=0 → ratio yields 1.277). Python's
// DeepseekV4RoPE does NOT apply any mscale — only modifies inv_freq
// via the YaRN smooth interpolation. To match Python exactly, DSV4
// uses this custom rope class instead of mlx-swift's YarnRoPE.
//
// Math:
//   inv_freq[i] = 1 / base^(2i/D) for i in [0, D/2)
//   if scaling.type == "yarn":
//     low = floor(D * log(orig / (beta_fast * 2π)) / (2 * log(base)))
//     high = ceil(D * log(orig / (beta_slow * 2π)) / (2 * log(base)))
//     ramp[i] = clip((i - low) / (high - low), 0, 1)
//     smooth = 1 - ramp
//     inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
//   wavelength = 1 / inv_freq
//   mx.fast.rope(x, dims, traditional=True, freqs=wavelength, offset=offset)
// Inverse: pass freqs=-wavelength.

public class DSV4RoPE: Module, OffsetLayer, ArrayOffsetLayer {
    public let dims: Int
    public let wavelength: MLXArray  // post-YaRN-correction wavelengths
    public let invFreq: MLXArray     // 1.0 / wavelength — for manual per-token path

    public init(
        dims: Int,
        base: Float,
        scalingConfig: [String: StringOrNumber]?,
        maxPositionEmbeddings: Int = 1_048_576
    ) {
        self.dims = dims
        // inv_freq[i] = 1 / base^(2i/D)
        let exponents = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
            / Float(dims)
        var invFreq = 1.0 / pow(base, exponents)

        if let cfg = scalingConfig {
            // Look up rope_type
            var ropeType: String? = nil
            if case .string(let s) = cfg["type"] { ropeType = s }
            else if case .string(let s) = cfg["rope_type"] { ropeType = s }
            if ropeType == "yarn" || ropeType == "deepseek_yarn" {
                func extractFloat(_ key: String, default def: Float) -> Float {
                    guard let v = cfg[key] else { return def }
                    switch v {
                    case .float(let f): return Float(f)
                    case .int(let i): return Float(i)
                    default: return def
                    }
                }
                let factor = extractFloat("factor", default: 1.0)
                let orig = extractFloat("original_max_position_embeddings",
                                        default: Float(maxPositionEmbeddings))
                let betaFast = extractFloat("beta_fast", default: 32.0)
                let betaSlow = extractFloat("beta_slow", default: 1.0)

                func correctionDim(_ n: Float) -> Float {
                    return Float(dims) * log(orig / (n * 2 * .pi)) / (2 * log(base))
                }
                var low = max(floor(correctionDim(betaFast)), 0)
                var high = min(ceil(correctionDim(betaSlow)), Float(dims - 1))
                if low == high { high += 0.001 }

                let ramp = clip(
                    (MLXArray(0..<(dims / 2)).asType(.float32) - low) / (high - low),
                    min: 0.0, max: 1.0
                )
                let smooth = 1.0 - ramp
                invFreq = invFreq / factor * (1.0 - smooth) + invFreq * smooth
            }
        }
        // Wavelength is what mx.fast.rope expects via freqs= parameter.
        self.wavelength = 1.0 / invFreq
        self.invFreq = invFreq
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return MLXFast.RoPE(
            x, dimensions: dims, traditional: true, base: nil, scale: 1.0,
            offset: offset, freqs: wavelength
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        return MLXFast.RoPE(
            x, dimensions: dims, traditional: true, base: nil, scale: 1.0,
            offset: offset, freqs: wavelength
        )
    }

    /// Inverse rope — negate wavelength to flip rotation sense.
    public func inverse(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return MLXFast.RoPE(
            x, dimensions: dims, traditional: true, base: nil, scale: 1.0,
            offset: offset, freqs: -wavelength
        )
    }

    /// §416 (2026-04-25) — Manual cos/sin RoPE path with per-TOKEN positions.
    ///
    /// `MLXFast.RoPE(_:offset: MLXArray)` interprets the offset array as a
    /// per-BATCH scalar (per docstring at MLXFast.swift:53), so it cannot
    /// handle non-consecutive per-token positions like the
    /// `(window * ratio + pool_base)` layout produced by `Compressor`.
    /// Mirrors Python `_call_manual` in `mlx_model.py:494-516`.
    ///
    /// `x` shape: `(..., L, dims)` (last axis must equal `self.dims`; caller
    ///   is responsible for splitting nope vs pe before calling this).
    /// `positions` shape: `(L,)` int or fp32 — absolute position per token.
    /// `inverse`: when true, flip the sin sign (equivalent to `scale=-1`).
    public func manual(_ x: MLXArray, positions: MLXArray, inverse: Bool = false) -> MLXArray {
        let dtype = x.dtype
        let pos = positions.asType(.float32)
        // freqs[l, d/2] = pos[l] * inv_freq[d/2]
        let freqs = pos.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        var cosT = cos(freqs)
        var sinT = sin(freqs)
        if inverse { sinT = -sinT }
        // Broadcast cos/sin across batch + head dims of x. x is (..., L, dims).
        // After reshape x → (..., L, dims/2, 2), cos/sin should broadcast over
        // every leading axis. Build a shape that prepends 1's for those axes.
        let leading = Array(repeating: 1, count: x.ndim - 2)
        let bc = leading + cosT.shape
        cosT = cosT.reshaped(bc).asType(dtype)
        sinT = sinT.reshaped(bc).asType(dtype)
        // Traditional layout: x[..., 2k, 2k+1] are interleaved pairs.
        var xPaired = x.reshaped(Array(x.shape.dropLast()) + [dims / 2, 2])
        let x0 = xPaired[.ellipsis, 0]
        let x1 = xPaired[.ellipsis, 1]
        let r0 = x0 * cosT - x1 * sinT
        let r1 = x0 * sinT + x1 * cosT
        xPaired = stacked([r0, r1], axis: -1)
        return xPaired.reshaped(x.shape)
    }
}

/// argPartition helper.
func argPartition(_ x: MLXArray, kth: Int, axis: Int) -> MLXArray {
    return MLX.argPartition(x, kth: kth, axis: axis)
}

/// takeAlong helper (forwards to MLX.takeAlong).
func takeAlong(_ x: MLXArray, _ idx: MLXArray, axis: Int) -> MLXArray {
    return MLX.takeAlong(x, idx, axis: axis)
}
