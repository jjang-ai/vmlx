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
    /// Legacy scalar alias for uniform-bit callers. For mixed projection
    /// profiles this equals `gateBits`; use the explicit fields below for
    /// runtime decisions.
    public let bits: Int
    public let gateBits: Int
    public let upBits: Int
    public let downBits: Int
    public let mxtqSeed: Int
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// SwiGLU clamp magnitude. `0.0` (default) preserves ordinary SwiGLU
    /// `silu(gate) * up` for every non-DSV4 caller — output is bit-identical
    /// to the pre-2026-05-04 path. DSV4 sets this to `10.0` to activate the
    /// limited-SwiGLU expression `silu(min(gate, 10)) * clip(up, -10, 10)`
    /// the Python `_dsv4_swiglu` reference uses and the codex_dsv4_fixkit
    /// Python runtime patch installs.
    public let swigluLimit: Float

    public init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        bits: Int = 2, seed: Int = 42,
        gateBits: Int? = nil,
        upBits: Int? = nil,
        downBits: Int? = nil,
        swigluLimit: Float = 0.0
    ) {
        let resolvedGateBits = gateBits ?? bits
        let resolvedUpBits = upBits ?? bits
        let resolvedDownBits = downBits ?? bits
        precondition(
            resolvedGateBits == resolvedUpBits,
            "TurboQuantSwitchGLU fused gate/up path requires gateBits == upBits; add a non-fused fallback before using mixed gate/up bits.")
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.bits = resolvedGateBits
        self.gateBits = resolvedGateBits
        self.upBits = resolvedUpBits
        self.downBits = resolvedDownBits
        self.mxtqSeed = seed
        self.swigluLimit = swigluLimit
        self._gateProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: resolvedGateBits, seed: seed
        )
        self._upProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: resolvedUpBits, seed: seed
        )
        self._downProj.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims, outFeatures: inputDims,
            numExperts: numExperts, bits: resolvedDownBits, seed: seed
        )
        super.init()
    }

    /// Cache of compiled MoE fast-path closures keyed by
    /// `(batchTokens, K, gateBits, upBits, downBits)`. Each closure
    /// captures the layer's in/out dimensions and projection bit widths
    /// and runs the full
    /// rotate → fused gate+up+SwiGLU → rotate → gather chain inside
    /// one `mx.compile(shapeless: true)` graph. This collapses 4
    /// individual Metal kernel dispatches per layer into a single
    /// command pipeline — same trick `_get_compiled_decode` does on
    /// the Python `load_jangtq` fast path. Empirical: ~3× decode tok/s
    /// on Qwen 3.6 JANGTQ_2L (M4 Max).
    private var compiledCache: [String: ([MLXArray]) -> [MLXArray]] = [:]

    private static let compiledFastPathEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_TQ_SWITCH_GLU_COMPILE"]?
            .lowercased()
        return !(raw == "0" || raw == "false" || raw == "off" || raw == "no")
    }()

    /// Forward through the JANGTQ MoE MLP fast path.
    /// `x` shape: `(batch, seq, hidden)`. `indices` shape: `(batch, seq, K)`.
    /// Returns `(batch, seq, K, hidden)` to match `SwitchGLU` semantics —
    /// caller multiplies by router scores and sums over the K dim.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        guard let signsIn = JANGTQRuntimeCache.shared.signs(inFeatures: inputDims, seed: mxtqSeed),
              let signsDn = JANGTQRuntimeCache.shared.signs(inFeatures: hiddenDims, seed: mxtqSeed),
              let cbGate  = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: gateBits),
              let cbDown  = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: downBits)
        else {
            fatalError("JANGTQ runtime sidecar not loaded — call JANGTQRuntimeCache.shared.loadSidecar(...) first")
        }

        let inputDims = self.inputDims
        let xSize = x.size
        let batchTokens = xSize / inputDims
        let xFlat = x.reshaped([batchTokens, inputDims])

        let K = indices.dim(-1)
        let idxFlat = indices.reshaped([-1]).asType(.uint32)

        // BUG3 DIAG: dump xRot for layer 0 first call
        if ProcessInfo.processInfo.environment["DSV4_DUMP_LOGITS"] == "1" {
            nonisolated struct _GTQ { static var fired = false }
            if !_GTQ.fired {
                _GTQ.fired = true
                let xRot0 = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)
                // Token 8 (last) of input
                let tok = batchTokens - 1  // last token
                let xPre8 = xFlat[tok..<(tok+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let xRot8 = xRot0[tok..<(tok+1), 0..<8].asType(.float32).flattened().asArray(Float.self)
                let xRotN = sqrt((xRot0 * xRot0).sum()).item(Float.self)
                let signsV = signsIn[0..<8].asType(.float32).flattened().asArray(Float.self)
                // packed[107, 0..<4] (expert 107, first row, first 4 packed cols)
                let pck107 = gateProj.packed[107..<108, 0..<1, 0..<4].asType(.uint32).flattened().asArray(UInt32.self)
                let nrm107 = gateProj.norms[107..<108, 0..<8].asType(.float32).flattened().asArray(Float.self)
                let cb = cbGate.asType(.float32).flattened().asArray(Float.self)
                let msg = "[BUG3-TQ] xFlat[\(tok),:8]=\(xPre8)\n[BUG3-TQ] signs[:8]=\(signsV)\n[BUG3-TQ] xRot[\(tok),:8]=\(xRot8) full_norm=\(xRotN)\n[BUG3-TQ] gateProj.packed[107,0,:4]=\(pck107)\n[BUG3-TQ] gateProj.norms[107,:8]=\(nrm107)\n[BUG3-TQ] codebook=\(cb)\n"
                if let d = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: d) }
            }
        }

        if !Self.compiledFastPathEnabled {
            let xRot = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)
            let xAct = JANGTQKernels.fusedGateUpSwiGLU(
                xRot: xRot,
                packedGate: gateProj.packed, normsGate: gateProj.norms,
                packedUp: upProj.packed, normsUp: upProj.norms,
                codebook: cbGate, rhsIndices: idxFlat,
                batchTokens: batchTokens, K: K,
                inFeatures: inputDims, outFeatures: hiddenDims, bits: gateBits,
                swigluLimit: swigluLimit
            )
            let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)
            let y = JANGTQKernels.gatherTQ(
                xRot: xActRot,
                packed: downProj.packed, norms: downProj.norms,
                codebook: cbDown, rhsIndices: idxFlat,
                nRows: batchTokens * K,
                inFeatures: hiddenDims, outFeatures: inputDims, bits: downBits
            )
            var outShape = indices.shape
            outShape.append(inputDims)
            return y.reshaped(outShape).asType(x.dtype)
        }

        let cacheKey = "bt\(batchTokens).K\(K).gb\(gateBits).ub\(upBits).db\(downBits)"
        if compiledCache[cacheKey] == nil {
            // Capture dimensions in the closure; signs/codebooks come
            // from the runtime cache and don't need to be inputs (they
            // never change for a given (in_features, seed/bits) tuple).
            let inDim = self.inputDims
            let outDim = self.hiddenDims
            let gateBitsLocal = self.gateBits
            let downBitsLocal = self.downBits
            let bt = batchTokens
            let kLocal = K
            // §426 — capture swiglu_limit for the compiled body. DSV4
            // sets this to 10.0 to engage the kernel's limited-SwiGLU
            // branch; everyone else passes 0.0 → kernel skips clamp +
            // emits byte-identical output to the pre-§426 path.
            let swigluLimitLocal = self.swigluLimit
            let body: ([MLXArray]) -> [MLXArray] = { args in
                // args: [xFlat, packedGate, normsGate, packedUp, normsUp,
                //        packedDown, normsDown, signsIn, signsDn,
                //        codebookGate, codebookDown, idxFlat]
                let xR = JANGTQKernels.hadamardRotate(args[0], signs: args[7], dim: inDim)
                let xAct_ = JANGTQKernels.fusedGateUpSwiGLU(
                    xRot: xR,
                    packedGate: args[1], normsGate: args[2],
                    packedUp: args[3], normsUp: args[4],
                    codebook: args[9], rhsIndices: args[11],
                    batchTokens: bt, K: kLocal,
                    inFeatures: inDim, outFeatures: outDim, bits: gateBitsLocal,
                    swigluLimit: swigluLimitLocal
                )
                let xActR = JANGTQKernels.hadamardRotate(xAct_, signs: args[8], dim: outDim)
                let yLocal = JANGTQKernels.gatherTQ(
                    xRot: xActR,
                    packed: args[5], norms: args[6],
                    codebook: args[10], rhsIndices: args[11],
                    nRows: bt * kLocal,
                    inFeatures: outDim, outFeatures: inDim, bits: downBitsLocal
                )
                return [yLocal]
            }
            // shapeless: true so the same compiled graph handles different
            // tokens-per-call without recompiling, mirroring Python.
            // BUG3 (2026-04-25): the compile path was earlier suspected as
            // the source of routed-expert divergence. Verified clean —
            // actual root cause was `resolveQuantOverrides` corrupting
            // `jangtqRoutedBits` from `quantization.bits=8`. Compile path
            // is fine, restore.
            compiledCache[cacheKey] = compile(shapeless: true, body)
        }
        let compiled = compiledCache[cacheKey]!

        let inputs: [MLXArray] = [
            xFlat,
            gateProj.packed, gateProj.norms,
            upProj.packed,   upProj.norms,
            downProj.packed, downProj.norms,
            signsIn, signsDn,
            cbGate, cbDown,
            idxFlat,
        ]
        let outputs = compiled(inputs)
        let y = outputs[0]

        var outShape = indices.shape
        outShape.append(inputDims)
        return y.reshaped(outShape).asType(x.dtype)
    }
}
