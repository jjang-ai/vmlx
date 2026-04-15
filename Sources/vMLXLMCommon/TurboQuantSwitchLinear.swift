//
// TurboQuantSwitchLinear — drop-in replacement for `SwitchLinear` that uses
// the JANGTQ codebook+Hadamard Metal kernels instead of `gather_qmm`.
// Created by Jinho Jang (eric@jangq.ai).
//
// Storage:
//   - `packed`  : uint32, shape (n_experts, out_features, packed_in)
//                 — codebook indices, 16 vals × 2 bits per uint32
//   - `norms`   : fp16,   shape (n_experts, out_features)
//                 — per-row L2 norm
//   - `signs`   : fp32,   shape (in_features,)
//                 — Hadamard sign vector (loaded from sidecar)
//   - `codebook`: fp32,   shape (4,)  for 2-bit
//                 — Lloyd-Max centroids (loaded from sidecar)
//
// `signs` and `codebook` are NOT module parameters — they're cached at
// load time in `JANGTQRuntimeCache` so multiple layers with the same
// `in_features` share the same MLXArray.
//
// `forward(x, indices)` does:
//   1. Hadamard rotate `x` (with `signs`) → `x_rot`  [P3 multiblock]
//   2. ONE Metal dispatch for the weighted dot products through the
//      codebook lookup, exactly mirroring `gather_qmm` semantics.
//
// For SwiGLU MoE blocks (gate+up+down), the higher-level
// `TurboQuantSwitchGLU` chains three of these via the fused gate+up
// kernel and the gather kernel. See `TurboQuantSwitchGLU` below.
//

import Foundation
import MLX
import MLXNN

/// Backed by the JANGTQ codebook kernels. Single matmul per call; no fused
/// gate+up. Use `TurboQuantSwitchGLU` for the full SwiGLU path.
public class TurboQuantSwitchLinear: Module {
    @ParameterInfo(key: "tq_packed") public var packed: MLXArray
    @ParameterInfo(key: "tq_norms")  public var norms: MLXArray

    public let inFeatures: Int
    public let outFeatures: Int
    public let numExperts: Int
    public let bits: Int
    public let mxtqSeed: Int

    public init(
        inFeatures: Int, outFeatures: Int, numExperts: Int,
        bits: Int = 2, seed: Int = 42
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.numExperts = numExperts
        self.bits = bits
        self.mxtqSeed = seed
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        // Initialize with zeros — the loader will overwrite with real data.
        self._packed.wrappedValue = MLXArray.zeros([numExperts, outFeatures, packedCols], dtype: .uint32)
        self._norms.wrappedValue  = MLXArray.zeros([numExperts, outFeatures], dtype: .float16)
        super.init()
    }

    /// Single-matmul forward (gate-only or up-only or down-only). For the
    /// fused gate+up+SwiGLU + down path, use `TurboQuantSwitchGLU` which
    /// dispatches the two specialized kernels in one chain.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Look up signs + codebook from the runtime cache.
        guard let signs = JANGTQRuntimeCache.shared.signs(inFeatures: inFeatures, seed: mxtqSeed)
        else {
            fatalError("JANGTQ runtime sidecar not loaded for inFeatures=\(inFeatures), seed=\(mxtqSeed)")
        }
        guard let codebook = JANGTQRuntimeCache.shared.codebook(inFeatures: inFeatures, bits: bits)
        else {
            fatalError("JANGTQ codebook missing for inFeatures=\(inFeatures), bits=\(bits)")
        }

        // Hadamard rotate input — accepts shape (..., in_features), returns fp32.
        let xRot = JANGTQKernels.hadamardRotate(x, signs: signs, dim: inFeatures)

        // Reshape to (batch, in_features) for the kernel.
        let batch = xRot.size / inFeatures
        let xFlat = xRot.reshaped([batch, inFeatures])

        // Number of expert slots K (last dim of indices)
        let K = indices.dim(-1)
        let idxFlat = indices.reshaped([-1]).asType(.uint32)

        // Use the gather kernel for the single matmul case (per-row mode).
        let y = JANGTQKernels.gatherTQ(
            xRot: xFlat, packed: packed, norms: norms,
            codebook: codebook, rhsIndices: idxFlat,
            nRows: batch * K, inFeatures: inFeatures, outFeatures: outFeatures, bits: bits
        )
        // Reshape output to match gather_qmm's `(..., K, 1, out_features)` shape
        // expected by callers (broadcast K).
        return y.reshaped(indices.shape + [1, outFeatures])
    }
}

/// Drop-in replacement for `SwitchGLU` that uses JANGTQ kernels for the
/// three projections. Mirrors the Python `_fused_switchglu_call` fast path
/// from `jang-tools/jang_tools/load_jangtq.py`.
public class TurboQuantSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: TurboQuantSwitchLinear
    @ModuleInfo(key: "up_proj")   public var upProj:   TurboQuantSwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: TurboQuantSwitchLinear

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let bits: Int
    public let mxtqSeed: Int

    public init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        bits: Int = 2, seed: Int = 42
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.bits = bits
        self.mxtqSeed = seed
        self._gateProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: bits, seed: seed
        )
        self._upProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: bits, seed: seed
        )
        self._downProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims, outFeatures: inputDims,
            numExperts: numExperts, bits: bits, seed: seed
        )
        super.init()
    }

    /// Forward through the JANGTQ MoE MLP fast path.
    /// `x` shape: `(batch, seq, hidden)`. `indices` shape: `(batch, seq, K)`.
    /// Returns `(batch, seq, K, hidden)` to match `SwitchGLU` semantics —
    /// caller multiplies by router scores and sums over the K dim.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        guard let signsIn = JANGTQRuntimeCache.shared.signs(inFeatures: inputDims, seed: mxtqSeed),
              let signsDn = JANGTQRuntimeCache.shared.signs(inFeatures: hiddenDims, seed: mxtqSeed),
              let cbGate  = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: bits),
              let cbDown  = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: bits)
        else {
            fatalError("JANGTQ runtime sidecar not loaded — call JANGTQRuntimeCache.shared.loadSidecar(...) first")
        }

        // The decode broadcast pattern: x has shape (batch, seq, hidden),
        // indices has shape (batch, seq, K). Each token uses K experts.
        // We flatten (batch, seq) → 1 batch row for the kernel.
        let inputDims = self.inputDims
        let xSize = x.size
        let batchTokens = xSize / inputDims
        let xFlat = x.reshaped([batchTokens, inputDims])

        let K = indices.dim(-1)
        let idxFlat = indices.reshaped([-1]).asType(.uint32)

        // 1. Rotate input
        let xRot = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)

        // 2. Fused gate+up+SwiGLU — broadcast mode: K_meta = K so the kernel
        //    can compute token_idx = dispatch_idx / K and k_idx = dispatch_idx % K.
        //    Total dispatches = batchTokens * K.
        let xAct = JANGTQKernels.fusedGateUpSwiGLU(
            xRot: xRot,
            packedGate: gateProj.packed, normsGate: gateProj.norms,
            packedUp: upProj.packed, normsUp: upProj.norms,
            codebook: cbGate, rhsIndices: idxFlat,
            batchTokens: batchTokens, K: K,
            inFeatures: inputDims, outFeatures: hiddenDims, bits: bits
        )
        // xAct shape: (batchTokens * K, hidden_dims)

        // 3. Hadamard rotate x_act — one row per (token, expert) pair.
        let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)

        // 4. Gather TQ matmul (down_proj) — per-row mode, one row per pair.
        let y = JANGTQKernels.gatherTQ(
            xRot: xActRot,
            packed: downProj.packed, norms: downProj.norms,
            codebook: cbDown, rhsIndices: idxFlat,
            nRows: batchTokens * K,
            inFeatures: hiddenDims, outFeatures: inputDims, bits: bits
        )
        // y shape: (batchTokens * K, inputDims)

        // Reshape to match SwitchGLU's output: (batch, seq, K, inputDims)
        var outShape = indices.shape
        outShape.append(inputDims)
        return y.reshaped(outShape).asType(x.dtype)
    }
}
