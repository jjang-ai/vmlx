//
// JANGTQDenseLinear — drop-in replacement for `MLXNN.Linear` whose weight
// matrix has been TurboQuant-codebook-quantized (.tq_packed + .tq_norms
// keys instead of `.weight`).
//
// Companion to `TurboQuantSwitchLinear` (MoE expert variant). This class
// targets the dense (non-MoE) decoder layers of Mistral 3 / Mistral 3.5 /
// Mistral 4 / Laguna JANGTQ bundles, where the entire text decoder
// (attention Q/K/V/O + MLP gate/up/down) is JANGTQ-quantized rather
// than just routed-expert MLPs.
//
// Storage (matches `jang_tools/turboquant/linear.tq_quantize_weight`,
// 2D shape because there's no expert dim — different from MoE):
//
//   - `tq_packed` : uint32, shape (out_features, packed_in)
//                   — codebook indices, 32/bits values per uint32
//   - `tq_norms`  : fp16,   shape (out_features,)
//                   — per-row L2 norm
//   - `signs`     : fp32,   shape (in_features,)
//                   — Hadamard sign vector — NOT a module parameter,
//                     loaded via `JANGTQRuntimeCache`
//   - `codebook`  : fp32,   shape (1<<bits,)
//                   — Lloyd-Max centroids — NOT a module parameter
//
// `forward(x)` does:
//   1. Hadamard rotate `x` (with `signs`) → `x_rot`
//   2. ONE Metal dispatch through the codebook lookup, mirroring the
//      semantics `Linear(W) @ x` would produce for the original weight.
//
// Implementation strategy: reuses the existing `JANGTQKernels.gatherTQ`
// metal kernel by reshaping the 2D `tq_packed` to 3D `[1, out, packed]`
// and feeding a zero-index array as the expert selector. The kernel's
// per-row mode handles `n_experts=1` correctly — singleton expert dim
// degenerates the gather to a regular row-of-codebook lookup.
//
// Bias support: matches `MLXNN.Linear`. If the source weight had a bias,
// it's stored as `.biases` (fp16 or fp32) and added after the matmul.
//
// Ported from vmlx-swift-lm/Libraries/MLXLMCommon/JANGTQDenseLinear.swift
// 2026-05-01 to bring up Mistral 3 / Mistral 3.5 (ministral3) / Laguna /
// Mistral 4 JANGTQ bundles in vMLX, which previously crashed at factory
// time because the dense JANGTQ port wasn't present.

import Foundation
import MLX
import MLXNN

public class JANGTQDenseLinear: Module {
    @ParameterInfo(key: "tq_packed") public var packed: MLXArray
    @ParameterInfo(key: "tq_norms")  public var norms: MLXArray
    @ParameterInfo(key: "biases")    public var biases: MLXArray?

    public let inFeatures: Int
    public let outFeatures: Int
    public let bits: Int
    public let mxtqSeed: Int
    public let hasBias: Bool

    public init(
        inFeatures: Int,
        outFeatures: Int,
        bits: Int = 2,
        seed: Int = 42,
        bias: Bool = false
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.bits = bits
        self.mxtqSeed = seed
        self.hasBias = bias
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        // Initialize with zeros — the loader overwrites with real weights.
        // Note the 2D shape (no expert dim) — matches what
        // `tq_quantize_weight` emits in the converter for dense layers.
        self._packed.wrappedValue = MLXArray.zeros(
            [outFeatures, packedCols], dtype: .uint32)
        self._norms.wrappedValue  = MLXArray.zeros(
            [outFeatures], dtype: .float16)
        if bias {
            self._biases.wrappedValue = MLXArray.zeros([outFeatures], dtype: .float32)
        } else {
            self._biases.wrappedValue = nil
        }
        super.init()
    }

    /// Forward through the JANGTQ codebook lookup. Reshape semantics:
    ///   - input  `x`:        any leading shape × inFeatures
    ///   - output `y`:        same leading shape × outFeatures
    ///
    /// Internally uses `JANGTQKernels.gatherTQ` with a singleton expert
    /// dim (n_experts=1) and a zero-index array as the expert selector.
    /// The kernel's per-row mode reduces the gather to a plain
    /// codebook-indexed matmul under that configuration.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let signs = JANGTQRuntimeCache.shared.signs(
            inFeatures: inFeatures, seed: mxtqSeed)
        else {
            fatalError(
                "JANGTQ runtime sidecar not loaded for inFeatures=\(inFeatures), "
                + "seed=\(mxtqSeed). Call `JANGTQRuntimeCache.shared.loadSidecar(...)` "
                + "before the first forward pass.")
        }
        guard let codebook = JANGTQRuntimeCache.shared.codebook(
            inFeatures: inFeatures, bits: bits)
        else {
            fatalError(
                "JANGTQ codebook missing for inFeatures=\(inFeatures), bits=\(bits)")
        }

        // Hadamard rotate input — accepts shape (..., in_features), returns fp32.
        let xRot = JANGTQKernels.hadamardRotate(x, signs: signs, dim: inFeatures)

        // Flatten leading dims for the kernel.
        let leadingShape = Array(x.shape.dropLast())
        let batch = xRot.size / inFeatures
        let xFlat = xRot.reshaped([batch, inFeatures])

        // Promote 2D packed → 3D with singleton expert dim. The gather
        // kernel needs n_experts in the leading dim; a singleton
        // degenerates the gather to a per-row codebook lookup.
        let packed3D = packed.reshaped([1, outFeatures, packed.dim(-1)])
        let norms2D = norms.reshaped([1, outFeatures])

        // Zero-index array: every row gathers expert 0 (the only one).
        let rhsIndices = MLXArray.zeros([batch], dtype: .uint32)

        let y = JANGTQKernels.gatherTQ(
            xRot: xFlat,
            packed: packed3D,
            norms: norms2D,
            codebook: codebook,
            rhsIndices: rhsIndices,
            nRows: batch,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            bits: bits
        )

        // Restore leading shape: (..., outFeatures)
        var out = y.reshaped(leadingShape + [outFeatures])

        // Restore output dtype to match input — kernel produces fp32.
        if x.dtype != .float32 {
            out = out.asType(x.dtype)
        }

        // Optional bias add, matching MLXNN.Linear semantics.
        if hasBias, let b = biases {
            out = out + b.asType(out.dtype)
        }

        return out
    }
}
