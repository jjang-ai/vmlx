//
// JANGTQ Metal kernels — Swift port of jang-tools/jang_tools/turboquant.
// Created by Jinho Jang (eric@jangq.ai).
//
// The kernel source strings here are byte-identical to the Python kernels
// that have been validated end-to-end on MiniMax M2.7 JANGTQ_2L:
//
//   ../../../../jang/jang-tools/jang_tools/turboquant/hadamard_kernel.py
//   ../../../../jang/jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py
//   ../../../../jang/jang-tools/jang_tools/turboquant/gather_tq_kernel.py
//
// Because we use `MLXFast.metalKernel(...)` (which calls the same
// `mlx_fast_metal_kernel_*` C++ entry points as Python's `mx.fast.metal_kernel`),
// the compiled Metal pipeline is BIT-IDENTICAL to the Python runtime.
//
// What this gives us:
//   - Decode speed exactly matches Python `mlx_lm` for JANGTQ models, because
//     every other op (attention, RMSNorm, RoPE, SDPA, lm_head) already uses
//     vendored mlx-swift kernels which are the same C++ kernels as Python MLX.
//   - All optimizations from the Python side (P3 multi-block Hadamard,
//     P12/P17 thread tiling, P9 vectorized unpack, P15 compile-friendly
//     wrappers) are preserved in the kernel source.
//
// Sweet-spot tile constants (P17, M3 Ultra sweep):
//   - jangtq_fused_gate_up_swiglu : OPT = 10 outputs per thread
//   - jangtq_gather_tq_matmul     : OPT = 20 outputs per thread
//

import Foundation
import MLX

// MARK: - Hadamard multiblock

private let kHadamardMultiblockSource = """
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint threads_per_tg = threads_per_threadgroup.x;

    uint total_d = meta[0];
    uint n_blocks = meta[1];

    threadgroup float shmem[4096];

    for (uint i = tid; i < total_d; i += threads_per_tg) {
        shmem[i] = static_cast<float>(x[batch_idx * total_d + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint offset = 0;
    for (uint b = 0; b < n_blocks; b++) {
        uint d_b = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        uint ept = (d_b + threads_per_tg - 1u) / threads_per_tg;
        if (ept == 0u) ept = 1u;

        for (uint stage = 0; stage < log_b; stage++) {
            uint h = 1u << stage;
            uint two_h = 2u * h;

            float newv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    uint block_start = (i_local / two_h) * two_h;
                    uint pos = i_local - block_start;
                    float a = shmem[offset + block_start + pos];
                    if (pos < h) {
                        newv[k] = a + shmem[offset + block_start + pos + h];
                    } else {
                        newv[k] = shmem[offset + block_start + pos - h] - a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    shmem[offset + i_local] = newv[k];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = 1.0f / sqrt(static_cast<float>(d_b));
        for (uint k = 0; k < ept; k++) {
            uint i_local = tid * ept + k;
            if (i_local < d_b) {
                out[batch_idx * total_d + offset + i_local] = shmem[offset + i_local] * norm;
            }
        }
        offset += d_b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

// MARK: - Fused gate+up+SwiGLU (P17 OPT=10)

private let kFusedSwiGLUSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 10u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc_g[10];
    float acc_u[10];
    #pragma unroll
    for (uint o = 0; o < 10; o++) { acc_g[o] = 0.0f; acc_u[o] = 0.0f; }

    uint expert_base = expert * out_features * packed_cols;
    uint x_off = token_idx * in_features;

    uint n_outs = 10u;
    if (out_idx_0 + 10u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[10], pvu[10];
        #pragma unroll
        for (uint o = 0; o < 10; o++) {
            if (o < n_outs) {
                uint row_off = expert_base + (out_idx_0 + o) * packed_cols + pack_idx;
                pvg[o] = packed_gate[row_off];
                pvu[o] = packed_up[row_off];
            } else {
                pvg[o] = 0u;
                pvu[o] = 0u;
            }
        }

        #pragma unroll
        for (uint k = 0; k < 16; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_off + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 10; o++) {
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 10; o++) {
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate[expert * out_features + oi]);
            float nu = static_cast<float>(norms_up[expert * out_features + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            out_act[base_off + oi] = (gv / (1.0f + metal::fast::exp(-gv))) * uv;
        }
    }
"""

// MARK: - Gather TQ matmul (P17 OPT=20)

private let kGatherTQSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint expert_base = expert * out_features * packed_cols;
    uint x_offset = token_idx * in_features;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;
        uint pv[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            pv[o] = (o < n_outs) ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
        }
        #pragma unroll
        for (uint k = 0; k < 16; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_offset + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                float w = codebook[(pv[o] >> shift) & mask];
                acc[o] += xv * w;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 20; o++) {
        acc[o] = simd_sum(acc[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[expert * out_features + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }
    }
"""

// MARK: - Public kernel access

/// Lazy-built singleton kernels. Each kernel is compiled once via
/// `MLXFast.metalKernel(...)` and cached for the lifetime of the process.
public enum JANGTQKernelLibrary {

    public static let hadamardMultiblock: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_hadamard_multiblock",
        inputNames: ["x", "signs", "meta"],
        outputNames: ["out"],
        source: kHadamardMultiblockSource
    )

    public static let fusedGateUpSwiGLU: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_fused_gate_up_swiglu",
        inputNames: [
            "x_rot", "packed_gate", "norms_gate",
            "packed_up", "norms_up",
            "codebook", "rhs_indices", "meta",
        ],
        outputNames: ["out_act"],
        source: kFusedSwiGLUSource
    )

    public static let gatherTQ: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_matmul",
        inputNames: ["x_rot", "packed", "norms", "codebook", "rhs_indices", "meta"],
        outputNames: ["out"],
        source: kGatherTQSource
    )
}

// MARK: - Codebook + signs cache

/// Sign and codebook arrays are deterministic functions of (in_features, seed/bits)
/// computed at quantization time via NumPy PCG64 + Lloyd-Max iteration. They're
/// loaded once at model load from `jangtq_runtime.safetensors` and cached here
/// keyed on `(in_features, seed)` / `(in_features, bits)`.
public final class JANGTQRuntimeCache: @unchecked Sendable {
    public static let shared = JANGTQRuntimeCache()

    private var signsByKey: [String: MLXArray] = [:]
    private var codebookByKey: [String: MLXArray] = [:]
    private let lock = NSLock()

    private init() {}

    public func loadSidecar(from sidecarPath: URL) throws {
        let loaded = try MLX.loadArrays(url: sidecarPath)
        lock.lock()
        defer { lock.unlock() }
        for (name, arr) in loaded {
            if name.hasPrefix("signs.") {
                signsByKey[name] = arr
            } else if name.hasPrefix("codebook.") {
                codebookByKey[name] = arr
            }
        }
    }

    public func signs(inFeatures: Int, seed: Int) -> MLXArray? {
        lock.lock(); defer { lock.unlock() }
        return signsByKey["signs.\(inFeatures).\(seed)"]
    }

    public func codebook(inFeatures: Int, bits: Int) -> MLXArray? {
        lock.lock(); defer { lock.unlock() }
        return codebookByKey["codebook.\(inFeatures).\(bits)"]
    }
}

// MARK: - High-level kernel wrappers (mirror Python `make_*_decode` factories)

public enum JANGTQKernels {

    /// Decompose a non-pow2 dim into a sum of pow2 blocks (largest first).
    public static func decomposePow2(_ dim: Int) -> [Int] {
        var blocks: [Int] = []
        var rem = dim
        while rem > 0 {
            let p = 1 << (Int.bitWidth - 1 - rem.leadingZeroBitCount)
            blocks.append(p)
            rem -= p
        }
        return blocks
    }

    /// Build the `meta` array the multiblock Hadamard kernel expects:
    /// `[total_d, n_blocks, d_b0, log_b0, d_b1, log_b1, ...]`
    public static func makeHadamardMeta(totalDim: Int) -> MLXArray {
        let blocks = decomposePow2(totalDim)
        var meta: [UInt32] = [UInt32(totalDim), UInt32(blocks.count)]
        for d in blocks {
            meta.append(UInt32(d))
            meta.append(UInt32(d.trailingZeroBitCount))
        }
        return MLXArray(meta)
    }

    /// Hadamard rotate `x` (any batch shape with `dim` last). Returns fp32.
    /// `signs` must be shape `(dim,)` fp32.
    public static func hadamardRotate(_ x: MLXArray, signs: MLXArray, dim: Int) -> MLXArray {
        // Flatten leading dims into batch
        var xFlat = x.reshaped([-1, dim]).asType(.float32)
        let batch = xFlat.shape[0]
        let meta = makeHadamardMeta(totalDim: dim)
        let blocks = decomposePow2(dim)
        let largestBlock = blocks.max() ?? dim
        let tgSize = min(1024, max(32, largestBlock))
        let outArrs = JANGTQKernelLibrary.hadamardMultiblock(
            [xFlat, signs, meta],
            template: nil,
            grid: (tgSize, batch, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [[batch, dim]],
            outputDTypes: [.float32]
        )
        var rot = outArrs[0]
        // Restore leading shape
        if x.ndim > 2 || (x.ndim == 2 && x.dim(0) != batch) {
            rot = rot.reshaped(x.shape)
        }
        return rot
    }

    /// Fused gate+up+SwiGLU.
    /// - `K` : experts per token (e.g. 8) — becomes `meta[0]` inside the kernel
    ///         so the kernel can compute `token_idx = dispatch_idx / K`.
    /// - `batchTokens` : number of input rows in `xRot` (tokens in the batch).
    ///         Total dispatches in `y` grid = `batchTokens * K`.
    /// - `xRot` shape: `(batchTokens, inFeatures)`
    /// - `rhsIndices` shape: `(batchTokens * K,)` uint32
    /// Returns fp32 of shape `(batchTokens * K, out_features)`.
    public static func fusedGateUpSwiGLU(
        xRot: MLXArray,
        packedGate: MLXArray, normsGate: MLXArray,
        packedUp: MLXArray,   normsUp: MLXArray,
        codebook: MLXArray,
        rhsIndices: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let meta = MLXArray([
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 10
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.fusedGateUpSwiGLU(
            [xRot, packedGate, normsGate, packedUp, normsUp,
             codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Gather TQ matmul in per-row mode (down_proj path).
    /// - `xRot` shape: `(nRows, inFeatures)` — one row per (token, expert) pair.
    /// - `rhsIndices` shape: `(nRows,)` uint32 — expert id for each row.
    /// Returns fp32 of shape `(nRows, outFeatures)`.
    public static func gatherTQ(
        xRot: MLXArray,
        packed: MLXArray, norms: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        nRows: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        // Per-row: K_meta = 1, so token_idx = dispatch_idx, k_idx = 0.
        let meta = MLXArray([
            UInt32(1), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQ(
            [xRot, packed, norms, codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nRows, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nRows, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }
}
