import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

// GELU approximate without the Power primitive (x ** 3). Uses x * x * x which
// decomposes to Multiply ops with proper output_shapes support.
// On M3+: compiled with compile(shapeless: true) for fused Metal dispatch.
// On M1/M2: runs as plain closure (compile(shapeless: true) crashes on Tahoe — MLX #3329).
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if HardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for MLXNN.GELU that avoids the Power primitive crash.
/// Use this anywhere `GELU(approximation: .precise)` or `.tanh` would be used.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

// Compiled activation kernels — fuses gate activation + element-wise multiply into
// a single Metal dispatch. Matches Python's @partial(mx.compile, shapeless=True).
// Guarded by HardwareInfo: M1/M2 + macOS Tahoe crashes with compile(shapeless: true).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, x: MLXArray) -> MLXArray in
        silu(gate) * x
    }
    if HardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, x: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * x
    }
    if HardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// iter-146 §216: public accessor for the compile-fused GELU-gate
/// activation used by Gemma-4 / Gemma-3 / Gemma-2 dense MLPs.
///
/// The private `compiledGeGLU` above is used internally by
/// `SwitchGLU` (the MoE expert dispatcher). Gemma4MLP and its
/// dense-family cousins re-implement the same `GELU(gate) * x`
/// pattern inline as two separate Metal dispatches — one GELU,
/// one element-wise multiply. This public wrapper lets those
/// sites route through the same fused kernel, saving one Metal
/// dispatch per decode token per layer. On a 62-layer dense
/// model (e.g. Gemma-4-31B) that's 62 fewer kernel launches
/// per token. Live-bench iter-144 showed Gemma-4-31B-JANG_4M
/// at 11 tok/s vs theoretical bandwidth cap ~32 tok/s, so
/// kernel-launch overhead is a plausible chunk of the gap.
///
/// Tracks §215 (stale-files Gemma4 hot-path port plan).
@Sendable
public func fusedGeGLU(_ gate: MLXArray, _ x: MLXArray) -> MLXArray {
    compiledGeGLU(gate, x)
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    // Fused gate+up gatherQuantizedMM cache. On decode (few-token batches),
    // concatenate gate_proj.weight and up_proj.weight along the output axis
    // once and dispatch a single wider kernel instead of two narrower ones.
    // Halves Metal dispatches for Qwen/MiniMax/Gemma4/GLM4 MoE decode.
    // Ported from vmlx-swift-lm @ 2026-04-15.
    private var fusedGateUpWeight: MLXArray? = nil
    private var fusedGateUpScales: MLXArray? = nil
    private var fusedGateUpBiases: MLXArray? = nil
    private var fusedGroupSize: Int = 64
    private var fusedBits: Int = 4
    private var fusedMode: QuantizationMode = .affine
    private var fusionAttempted: Bool = false

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        // Detect common activation types for compiled fast path.
        // Use safeGeluApproximate for comparison to avoid MLXNN's compiledGeluApproximate
        // which uses the Power primitive (x ** 3) and crashes on some Metal GPUs during
        // model load time — see comment on safeGeluApproximate above.
        let testInput = MLXArray([Float(1.0)])
        let testOutput = activation(testInput)
        let siluOutput = silu(testInput)
        let geluOutput = safeGeluApproximate(testInput)
        self.isSiluActivation = (testOutput .== siluOutput).all().item(Bool.self)
        self.isGeluActivation = !isSiluActivation && (testOutput .== geluOutput).all().item(Bool.self)

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Build the fused gate+up weight cache on first call. Guarded by
    /// `fusionAttempted` so it runs exactly once per SwitchGLU instance.
    /// Skipped via `BENCH_NO_FUSED_GATE_UP=1` for A/B. Only works when both
    /// projections are `QuantizedSwitchLinear` with matching quant params.
    private func ensureFusedGateUp() {
        if fusionAttempted { return }
        fusionAttempted = true
        if ProcessInfo.processInfo.environment["BENCH_NO_FUSED_GATE_UP"] == "1" { return }
        guard let g = gateProj as? QuantizedSwitchLinear,
              let u = upProj as? QuantizedSwitchLinear,
              g.groupSize == u.groupSize,
              g.bits == u.bits,
              g.mode == u.mode
        else { return }
        let fusedW = concatenated([g.weight, u.weight], axis: -2)
        let fusedS = concatenated([g.scales, u.scales], axis: -2)
        var fusedB: MLXArray? = nil
        if let gb = g.biases, let ub = u.biases {
            fusedB = concatenated([gb, ub], axis: -2)
        }
        // Force eager materialization so the first decode doesn't pay
        // the concat cost mid-generation. asyncEval schedules immediately
        // but doesn't block the caller, so we pay concat latency on the
        // GPU side in parallel with any other load-time work.
        asyncEval(fusedW)
        asyncEval(fusedS)
        if let fb = fusedB { asyncEval(fb) }
        self.fusedGateUpWeight = fusedW
        self.fusedGateUpScales = fusedS
        self.fusedGateUpBiases = fusedB
        self.fusedGroupSize = g.groupSize
        self.fusedBits = g.bits
        self.fusedMode = g.mode
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        ensureFusedGateUp()

        // Fused gate+up is a net WIN for decode (compute-bound, single wide
        // matmul has better GPU occupancy) and a net LOSS for prefill
        // (memory-bandwidth bound, two narrower matmuls have better cache
        // locality). Threshold 32 admits single-token + a few prompt tokens
        // as "decode-shaped" and bounces large prefill chunks to the
        // two-call path. Override via BENCH_FUSED_GATE_UP_THRESHOLD.
        let decodeThreshold: Int =
            Int(ProcessInfo.processInfo.environment["BENCH_FUSED_GATE_UP_THRESHOLD"] ?? "32") ?? 32
        let useFused = (fusedGateUpWeight != nil) && (indices.size <= decodeThreshold)

        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let activated: MLXArray
        if useFused, let fusedW = fusedGateUpWeight, let fusedS = fusedGateUpScales {
            // FUSED PATH — single gatherQuantizedMM for gate+up combined,
            // then split and apply compiled SwiGLU/GeGLU.
            let combined = MLX.gatherQuantizedMM(
                x, fusedW,
                scales: fusedS, biases: fusedGateUpBiases,
                rhsIndices: idx, transpose: true,
                groupSize: fusedGroupSize, bits: fusedBits, mode: fusedMode,
                sortedIndices: doSort)
            let splits = MLX.split(combined, parts: 2, axis: -1)
            let xGate = splits[0]
            let xUp = splits[1]
            if isSiluActivation {
                activated = compiledSwiGLU(xGate, xUp)
            } else if isGeluActivation {
                activated = compiledGeGLU(xGate, xUp)
            } else {
                activated = activation(xGate) * xUp
            }
        } else {
            // Fallback: two-call path for non-quantized models, prefill
            // batches above threshold, or feature-flag off.
            let xUp = upProj(x, idx, sortedIndices: doSort)
            let xGate = gateProj(x, idx, sortedIndices: doSort)
            if isSiluActivation {
                activated = compiledSwiGLU(xGate, xUp)
            } else if isGeluActivation {
                activated = compiledGeGLU(xGate, xUp)
            } else {
                activated = activation(xGate) * xUp
            }
        }
        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
