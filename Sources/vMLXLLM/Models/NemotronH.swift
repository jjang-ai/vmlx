//
//  NemotronH.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Block Type

private enum NemotronHBlockType {
    case mamba  // "M"
    case attention  // "*"
    case mlp  // "-"
    case moe  // "E"

    init(from char: Character) {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default: fatalError("Unknown NemotronH block type: \(char)")
        }
    }
}

// MARK: - Mixer Protocol

/// Protocol for all mixer types in NemotronH blocks
private protocol NemotronHMixer: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

// MARK: - Activations

/// Squared ReLU activation: relu(x)^2
///
/// Scalar zero is typed to `x.dtype` so `MLX.maximum` doesn't silently
/// promote the bfloat16/fp16 activations to fp32 on every call. The
/// untyped form costs ~400 AsType ops per decode step across the 26
/// Mamba+attention layers of Nemotron-Cascade-30B and was the dominant
/// remaining decode-time tax after TurboQuant KV was turned off
/// (reference commit `d4e4e45`: 45 → 110 tok/s on M4 Max).
private func relu2(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0, dtype: x.dtype))
    return y * y
}

// MARK: - MambaRMSNormGated

private class NemotronHRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int

    init(dimensions: Int, eps: Float, groupSize: Int) {
        self.eps = eps
        self.groupSize = groupSize
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        var states = x
        if let gate {
            states = states * silu(gate)
        }

        // Python: x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        // Reshape [..., hidden] -> [..., nGroups, groupSize] for per-group normalization
        let shape = states.shape
        var newShape = Array(shape.dropLast())
        newShape.append(-1)
        newShape.append(groupSize)
        let unflattened = states.reshaped(newShape)

        // Python: x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        // Apply RMS norm per group WITHOUT scaling (pass ones as weight)
        // Swift rmsNorm doesn't accept nil, so we use identity weight
        let normed = MLXFast.rmsNorm(unflattened, weight: MLXArray.ones([groupSize], dtype: unflattened.dtype), eps: eps)

        // Python: return self.weight * x.flatten(-2)
        // Flatten back to [..., hidden] and apply learned weight
        let flattened = normed.reshaped(shape)
        return weight * flattened
    }
}

// MARK: - Mamba2Mixer

private class NemotronHMamba2Mixer: Module, NemotronHMixer {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let headsPerGroup: Int

    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "norm") var norm: NemotronHRMSNormGated

    init(_ args: NemotronHConfiguration) {
        self.numHeads = args.mambaNumHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = args.ssmStateSize
        self.convKernelSize = args.convKernel
        self.intermediateSize = args.mambaNumHeads * args.mambaHeadDim
        self.numGroups = args.nGroups
        self.headDim = args.mambaHeadDim
        self.timeStepLimit = (args.timeStepLimitMin, args.timeStepLimitMax)
        self.headsPerGroup = numHeads / numGroups
        self.convDim = intermediateSize + 2 * numGroups * ssmStateSize

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        self._inProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: args.mambaProjBias)

        self._dtBias.wrappedValue = MLXArray.ones([numHeads])
        let headsRange = (MLXArray(0 ..< numHeads).asType(.float32) + 1)
        self._aLog.wrappedValue = MLX.log(headsRange)
        self._D.wrappedValue = MLXArray.ones([numHeads])

        let groupSize = intermediateSize / numGroups
        self._norm.wrappedValue = NemotronHRMSNormGated(
            dimensions: intermediateSize,
            eps: args.layerNormEpsilon,
            groupSize: groupSize
        )

        self._outProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: args.mambaProjBias)

        super.init()
    }

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        var convInput = input

        // Apply mask if present
        if let mask {
            let expandedMask = expandedDimensions(mask, axis: -1)
            convInput = MLX.where(expandedMask, convInput, MLXArray.zeros(like: convInput))
        }

        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]

        if convState == nil {
            if convKernelSize > 1 {
                convState = MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype)
            } else {
                convState = MLXArray.zeros([batch, 0, convDim], dtype: dtype)
            }
        }

        let padded = concatenated([convState!, convInput], axis: 1)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = padded[0..., start ..< end, 0...]
        }

        let convOutput = conv1d(padded)
        return silu(convOutput)
    }

    private func mambaForward(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let convOutput = applyConv(convInput, mask: mask, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )

        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        let previousState = cache?[1]
        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: previousState,
            timeStepLimit: timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = nextState
        }

        let flattenedY = y.flattened(start: 2)
        return outProj(norm(flattenedY, gate: gate))
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        mambaForward(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

// MARK: - Attention

private class NemotronHAttention: Module, NemotronHMixer {
    let args: NemotronHConfiguration
    let scale: Float
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    // NOTE: NemotronH attention does NOT use RoPE (unlike most transformer models)
    // The Python implementation at mlx_lm/models/nemotron_h.py lines 247-274 shows
    // direct attention without position embeddings

    init(_ args: NemotronHConfiguration) {
        self.args = args
        self.numHeads = args.numAttentionHeads
        self.numKeyValueHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let attentionBias = args.attentionBias

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: attentionBias)
        self._wk.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wv.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: attentionBias)

        super.init()
    }

    private func attentionForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let queries = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        // No RoPE applied - NemotronH attention uses direct attention without position embeddings

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        attentionForward(x, mask: attentionMask, cache: cache)
    }
}

// MARK: - MLP

private class NemotronHMLP: Module, UnaryLayer, NemotronHMixer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: NemotronHConfiguration, intermediateSize: Int? = nil) {
        let intermediate = intermediateSize ?? args.intermediateSize
        self._upProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: args.mlpBias)
        self._downProj.wrappedValue = Linear(intermediate, args.hiddenSize, bias: args.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(relu2(upProj(x)))
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - MoE Gate

private func groupExpertSelect(
    gates: MLXArray,
    eSCB: MLXArray,  // e_score_correction_bias
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    let (bsz, seqLen) = (gates.dim(0), gates.dim(1))

    // Original scores using sigmoid
    let origScores = sigmoid(gates)
    var scores = origScores + eSCB

    // Group-based selection if n_group > 1
    if nGroup > 1 {
        let numExperts = scores.dim(-1)
        let expertsPerGroup = numExperts / nGroup

        // Reshape to [batch, seq, n_group, experts_per_group]
        let groupScores = scores.reshaped(bsz, seqLen, nGroup, -1)

        // Get top-2 per group and sum for group scores (using sorted to get top values)
        let topKGroupScores = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(
            axis: -1, keepDims: true)

        // Keep only top topkGroup groups (zero out the rest)
        let k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, expertsPerGroup])

        // Zero out scores from non-selected groups
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0, dtype: groupScores.dtype), axis: -2)

        // Flatten back
        scores = flattened(scores, start: -2, end: -1)
    }

    // Unary `-` preserves dtype; scalar `MLXArray(0)` defaults fp32 and
    // force-promotes bf16 operands (AsType cascade). See §27 +
    // mlp_bfloat16_upcast.md.
    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var finalScores = takeAlong(origScores, inds, axis: -1)

    // Normalize if needed
    if topK > 1 && normTopkProb {
        let denominator = finalScores.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: finalScores.dtype)
        finalScores = finalScores / denominator
    }

    // Apply scaling factor
    finalScores = finalScores * routedScalingFactor

    return (inds, finalScores)
}

private class NemotronHMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eSCB: MLXArray

    init(_ args: NemotronHConfiguration) {
        self.topK = args.numExpertsPerTok
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb

        self._weight.wrappedValue = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        self._eSCB.wrappedValue = MLXArray.zeros([args.nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = MLX.matmul(x, weight.transposed())
        return groupExpertSelect(
            gates: gates,
            eSCB: eSCB,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

// MARK: - SwitchMLP for NemotronH (uses relu2 instead of silu/glu)

private class NemotronHSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        self._fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        self._fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size > 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        var y = fc1(x, idx, sortedIndices: doSort)
        y = relu2(y)
        y = fc2(y, idx, sortedIndices: doSort)

        if doSort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(y, axis: -2)
    }
}

// MARK: - §437 JANGTQ SwitchMLP for NemotronH (TurboQuant codec on routed experts)

/// Drop-in replacement for `NemotronHSwitchMLP` that uses TurboQuant
/// per-expert weights. Mirrors the affine path semantically:
///   y = fc2( relu²( fc1(x, idx) ), idx )
///
/// The TQ codec encodes each expert's weight matrix as a Hadamard-rotated
/// codebook lookup. `TurboQuantSwitchLinear` handles the per-expert
/// matmul + Hadamard rotate + codebook lookup in one Metal dispatch.
/// Activation stays as `relu²` between the two TQ matmuls — no fused
/// ReLU² kernel exists yet, but the split path is correct and runs on
/// the same gather_tq Metal kernel as MiniMax / DSV4 JANGTQ.
private class NemotronHJANGTQSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: TurboQuantSwitchLinear
    @ModuleInfo(key: "fc2") var fc2: TurboQuantSwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, bits: Int, seed: Int) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        self._fc1.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: bits, seed: seed)
        self._fc2.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims, outFeatures: inputDims,
            numExperts: numExperts, bits: bits, seed: seed)

        super.init()
    }

    /// Same shape contract as `NemotronHSwitchMLP.callAsFunction` —
    /// callers (NemotronHMoE / NemotronHJANGTQMoE) can't tell the
    /// difference. `expandedDimensions(_:axes:[-2,-3])` matches what
    /// TurboQuantSwitchLinear expects for the gather kernel input.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let xExp = MLX.expandedDimensions(x, axes: [-2, -3])
        var y = fc1(xExp, indices)
        y = relu2(y)
        y = fc2(y, indices)
        return MLX.squeezed(y, axis: -2)
    }
}

// MARK: - MoE

private class NemotronHMoE: Module, UnaryLayer, NemotronHMixer {
    let numExpertsPerTok: Int
    let hasSharedExperts: Bool

    @ModuleInfo(key: "gate") var gate: NemotronHMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronHSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHMLP?

    /// Flash MoE Phase 2b shim. The Nemotron expert path uses
    /// `SwitchLinear` (fc1/fc2, 2-proj relu²) not 3-proj GLU, so the
    /// installed block must be constructed with `isSwitchGLU=false`.
    /// Shared experts still run natively.
    fileprivate var flashMoeShim: FlashMoEBlock? = nil

    init(_ args: NemotronHConfiguration) {
        self.numExpertsPerTok = args.numExpertsPerTok
        self.hasSharedExperts = args.nSharedExperts != nil && args.nSharedExperts! > 0

        self._gate.wrappedValue = NemotronHMoEGate(args)
        self._switchMLP.wrappedValue = NemotronHSwitchMLP(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts
        )

        if hasSharedExperts {
            self._sharedExperts.wrappedValue = NemotronHMLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let shim = flashMoeShim {
            y = shim(x)
        } else {
            let (inds, scores) = gate(x)
            y = switchMLP(x, inds)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }

        if let sharedExperts {
            y = y + sharedExperts(x)
        }

        return y
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - Latent MoE (Nemotron-3-Super-120B)

/// Latent-projection MoE block. Port of
/// `vmlx_engine/utils/nemotron_latent_moe.py:LatentNemotronHMoE`.
///
/// Forward:
/// ```
/// inds, scores = gate(x)              // full hidden (4096)
/// x_latent = fc1_latent_proj(x)       // compress 4096 → 1024
/// y = switch_mlp(x_latent, inds)      // experts in latent (1024→2688→1024)
/// y = (y * scores[..., None]).sum(-2) // weighted sum
/// y = fc2_latent_proj(y)              // expand 1024 → 4096
/// y = y + shared_experts(x)           // shared expert in full dim
/// ```
///
/// The `fc1_latent_proj` / `fc2_latent_proj` weights load from the
/// shard directly by name — no sanitize() rewrites required. The
/// existing `switch_mlp.{down_proj,up_proj}` → `switch_mlp.{fc2,fc1}`
/// rename in the JANG loader path still applies to this variant
/// (the expert stack itself is unchanged in shape, just sitting in
/// latent space instead of full hidden).
///
/// Gate + shared expert run in FULL hidden_size (4096). Only the
/// routed experts compress through the latent bottleneck — which is
/// the whole point of the optimization: latent MoE cuts expert FLOPs
/// by `(hidden/latent)^2` without hurting routing quality.
private class NemotronHLatentMoE: Module, UnaryLayer, NemotronHMixer {
    let numExpertsPerTok: Int
    let hasSharedExperts: Bool

    @ModuleInfo(key: "gate") var gate: NemotronHMoEGate
    @ModuleInfo(key: "fc1_latent_proj") var fc1Latent: Linear
    @ModuleInfo(key: "fc2_latent_proj") var fc2Latent: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronHSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHMLP?

    init(_ args: NemotronHConfiguration) {
        self.numExpertsPerTok = args.numExpertsPerTok
        self.hasSharedExperts = args.nSharedExperts != nil && args.nSharedExperts! > 0

        // moeLatentSize is guaranteed non-nil by the caller
        // (`NemotronHBlock` branches on it before constructing us).
        guard let latentSize = args.moeLatentSize else {
            fatalError(
                "NemotronHLatentMoE constructed without moe_latent_size — " +
                "caller should branch on args.moeLatentSize before picking this variant."
            )
        }

        self._gate.wrappedValue = NemotronHMoEGate(args)

        // Shared projections — no bias, compression/expansion only.
        self._fc1Latent.wrappedValue = Linear(
            args.hiddenSize, latentSize, bias: false)
        self._fc2Latent.wrappedValue = Linear(
            latentSize, args.hiddenSize, bias: false)

        // Experts operate on latent-size tensors: both fc1 and fc2
        // inside the SwitchMLP use `inputDims = latentSize` so the
        // per-expert weight shapes are `[numExperts, moeIntermediateSize,
        // latentSize]` instead of `[numExperts, moeIntermediateSize,
        // hiddenSize]`. This is what `fc1_latent_proj` + `fc2_latent_proj`
        // compress into.
        self._switchMLP.wrappedValue = NemotronHSwitchMLP(
            inputDims: latentSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts
        )

        if hasSharedExperts {
            // Shared expert operates in FULL hidden_size — no latent
            // compression. The shared path is always-on for every
            // token regardless of routing, and its capacity is the
            // full hidden width.
            self._sharedExperts.wrappedValue = NemotronHMLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Route on full hidden (the gate sees the uncompressed
        // representation so it can pick experts based on full context).
        let (inds, scores) = gate(x)

        // Compress to latent space before expert computation.
        let xLatent = fc1Latent(x)

        // Expert matmul in latent space. Returns [..., K, latentSize].
        var y = switchMLP(xLatent, inds)

        // Weighted sum across the top-K expert axis.
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)

        // Expand back to full hidden before the residual add.
        y = fc2Latent(y)

        // Shared expert path: operates in full hidden on the original
        // input `x`, not on the latent-compressed version. This is
        // load-bearing — the shared expert is a dense MLP that needs
        // the full-dimensional hidden state to be useful.
        if let sharedExperts {
            y = y + sharedExperts(x)
        }

        return y
    }

    // Protocol conformance — mixer is dispatched via NemotronHMixer
    // inside NemotronHBlock's callAsFunction so the block type lookup
    // stays branchless on the hot path.
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - Decoder Block

private class NemotronHBlock: Module {
    let blockType: NemotronHBlockType

    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Single mixer property with base Module type - concrete type assigned at init
    // This pattern is used by Jamba for feedForward: Module
    @ModuleInfo(key: "mixer") var mixer: Module

    init(_ args: NemotronHConfiguration, blockType: Character) {
        self.blockType = NemotronHBlockType(from: blockType)

        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Assign the appropriate concrete mixer type.
        //
        // `.moe` branches on `args.moeLatentSize`:
        //   - nil       → standard NemotronHMoE (full-hidden experts)
        //   - Some(n)   → NemotronHLatentMoE (n-dim bottleneck around
        //                 the expert block, used by Nemotron-3-Super-120B)
        //
        // The Python side monkey-patches mlx-lm's `NemotronHBlock.__init__`
        // to swap the mixer after the fact; we own the source so we
        // branch at construction time and keep the mixer type stable
        // for the life of the block.
        switch self.blockType {
        case .mamba:
            _mixer.wrappedValue = NemotronHMamba2Mixer(args)
        case .attention:
            _mixer.wrappedValue = NemotronHAttention(args)
        case .mlp:
            _mixer.wrappedValue = NemotronHMLP(args)
        case .moe:
            if args.moeLatentSize != nil {
                _mixer.wrappedValue = NemotronHLatentMoE(args)
            } else {
                _mixer.wrappedValue = NemotronHMoE(args)
            }
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let hidden = norm(x)

        // Cast to protocol and call - all mixer types conform to NemotronHMixer
        let mixerFunc = mixer as! NemotronHMixer
        let output = mixerFunc(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)

        return x + output
    }
}

// MARK: - Backbone (matches Python's NemotronHModel which is stored as self.backbone)

private class NemotronHBackbone: Module {
    let args: NemotronHConfiguration

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    // Cache indices (into the cache list, not pattern indices)
    // Python: fa_idx counts Mamba layers before first Attention
    // Python: ssm_idx counts Attention layers before first Mamba
    let firstAttentionCacheIndex: Int?
    let firstMambaCacheIndex: Int?

    init(_ args: NemotronHConfiguration) {
        self.args = args
        precondition(args.vocabSize > 0)

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)

        let pattern = Array(args.hybridOverridePattern)
        self._layers.wrappedValue = pattern.map { NemotronHBlock(args, blockType: $0) }

        self._normF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Calculate cache indices (only Mamba + Attention have caches)
        // fa_idx: count Mamba layers (M) before the first Attention layer (*)
        var faIdx: Int? = nil
        var mambaCount = 0
        for char in pattern {
            if char == "*" {
                faIdx = mambaCount
                break
            } else if char == "M" {
                mambaCount += 1
            }
        }
        self.firstAttentionCacheIndex = faIdx

        // ssm_idx: count Attention layers (*) before the first Mamba layer (M)
        var ssmIdx: Int? = nil
        var attnCount = 0
        for char in pattern {
            if char == "M" {
                ssmIdx = attnCount
                break
            } else if char == "*" {
                attnCount += 1
            }
        }
        self.firstMambaCacheIndex = ssmIdx

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = embeddings(inputs)
        return forwardFromEmbeddings(hidden, cache: cache)
    }

    /// §436 — multimodal entry point. Skips the token embedding lookup
    /// and accepts pre-computed embeddings (text + spliced
    /// vision/video/audio). Mirrors `NemotronHOmni.prepare()` flow:
    /// the omni wrapper builds embeds for text, runs RADIO on image
    /// pixels, runs Parakeet on audio mel-spec, splices each at the
    /// matching context-token positions, then hands the merged tensor
    /// here.
    func forwardFromEmbeddings(_ inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = inputsEmbeds

        // Create attention mask using the first attention layer's cache
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let cacheIdx = firstAttentionCacheIndex,
                let cache = cache,
                cacheIdx < cache.count
            else { return .none }
            return createAttentionMask(h: hidden, cache: cache[cacheIdx])
        }()

        // Create SSM mask using the first Mamba layer's cache
        // Python: ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        let ssmMask: MLXArray? = {
            guard let cacheIdx = firstMambaCacheIndex,
                let cache = cache,
                cacheIdx < cache.count,
                let mambaCache = cache[cacheIdx] as? MambaCache
            else { return nil }
            return mambaCache.makeMask(N: hidden.dim(1))
        }()

        // Track which cache to use for each layer
        var cacheCounter = 0
        for layer in layers {
            let c: KVCache?
            if layer.blockType == .mamba || layer.blockType == .attention {
                c = cache?[cacheCounter]
                cacheCounter += 1
            } else {
                c = nil
            }

            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: c)
        }

        return normF(hidden)
    }
}

// MARK: - Main Model (matches Python's Model class)

public class NemotronHModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "backbone") private var backbone: NemotronHBackbone
    let configuration: NemotronHConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] {
        backbone.layers
    }

    public init(_ args: NemotronHConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabSize

        // kvHeads array: non-zero for attention layers, zero for others
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }

        self._backbone.wrappedValue = NemotronHBackbone(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        return out
    }

    /// §436 — Public token-embedding accessor. Used by `NemotronHOmni`
    /// to build a starter embedding tensor that the multimodal wrapper
    /// then splices vision / audio embeddings into at placeholder
    /// token positions before handing back to
    /// `callAsFunction(inputsEmbeds:cache:)`.
    public func embedTokens(_ tokens: MLXArray) -> MLXArray {
        backbone.embeddings(tokens)
    }

    /// §436 — Multimodal forward. Bypasses the token-embedding lookup
    /// and accepts pre-computed embeddings (text + RADIO vision +
    /// Parakeet audio merged via `spliceAtToken`). Same lm_head /
    /// tied-embedding logic as `callAsFunction(_:cache:)`.
    public func callAsFunction(inputsEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone.forwardFromEmbeddings(inputsEmbeds, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let pattern = Array(configuration.hybridOverridePattern)
        return pattern.compactMap { char -> KVCache? in
            let blockType = NemotronHBlockType(from: char)
            switch blockType {
            case .mamba:
                return MambaCache()
            case .attention:
                return KVCacheSimple()
            case .mlp, .moe:
                return nil  // No cache needed for MLP/MoE layers
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        // Filter MTP (multi-token prediction) and per-weight importance sidecar
        // entries that JANG exporters leave in Nemotron-H bundles. These keys
        // are not part of the runtime module tree — if left in, load_weights
        // rejects them as strict-mode mismatches and, worse, smelt-mode +
        // flash-moe cross-contaminate MTP weights into routed experts.
        // Mirrors Python `utils/jang_loader.py` _load_jang_v2 MTP filter
        // (`not k.endswith(".importance") and "mtp." not in k`).
        let filteredWeights = weights.filter { key, _ in
            !key.hasSuffix(".importance") && !key.contains("mtp.")
        }

        for (key, value) in filteredWeights {
            var finalValue = value

            // Handle conv1d weight axis swap
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                finalValue = value.swappedAxes(1, 2)
            }

            sanitized[key] = finalValue
        }

        // JANG models have pre-stacked expert weights as switch_mlp.{down,up}_proj
        // but the module tree expects switch_mlp.{fc2,fc1}. Remap key names.
        let jangExpertRenames: [(String, String)] = [
            (".switch_mlp.down_proj.", ".switch_mlp.fc2."),
            (".switch_mlp.up_proj.", ".switch_mlp.fc1."),
        ]
        var remapped = [String: MLXArray]()
        for (key, value) in sanitized {
            var newKey = key
            for (from, to) in jangExpertRenames {
                if newKey.contains(from) {
                    newKey = newKey.replacingOccurrences(of: from, with: to)
                }
            }
            remapped[newKey] = value
        }
        sanitized = remapped

        // Stack per-expert weights (HF format): backbone.layers.{l}.mixer.experts.{e}.{proj}.weight
        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(l).mixer"
            for (m, n) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                if sanitized["\(prefix).experts.0.\(m).weight"] != nil {
                    let toJoin = (0 ..< configuration.nRoutedExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(m).weight")
                    }
                    if !toJoin.isEmpty {
                        sanitized["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitized
    }

    /// Predicate for casting: some parameters should stay in float32
    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }
}

// MARK: - Configuration

public struct NemotronHConfiguration: Codable, Sendable {
    public var modelType: String = "nemotron_h"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionBias: Bool
    public var mambaNumHeads: Int
    public var mambaHeadDim: Int
    public var mambaProjBias: Bool
    public var ssmStateSize: Int
    public var convKernel: Int
    public var nGroups: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var moeSharedExpertIntermediateSize: Int
    public var nRoutedExperts: Int
    public var nSharedExperts: Int?
    public var numExpertsPerTok: Int
    /// Latent MoE compression size. When set, MoE blocks route through
    /// a two-stage projection — `fc1_latent_proj` compresses the hidden
    /// state to this size before expert computation, then `fc2_latent_proj`
    /// expands back. Nemotron-3-Super-120B uses this (hidden=4096 →
    /// latent=1024 → experts=1024→2688→1024 → hidden=4096). When nil,
    /// the MoE block uses the standard SwitchMLP path in full hidden
    /// dimension.
    ///
    /// Port of `vmlx_engine/utils/nemotron_latent_moe.py`. Python
    /// monkey-patches mlx-lm's `nemotron_h` module at runtime; Swift
    /// has the model source directly so we branch on this field inside
    /// `NemotronHBlock` to pick the right MoE variant at construction.
    public var moeLatentSize: Int?
    public var hybridOverridePattern: String
    public var layerNormEpsilon: Float
    public var mlpBias: Bool
    public var useBias: Bool
    public var useConvBias: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var headDim: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var timeStepLimitMin: Float
    public var timeStepLimitMax: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLatentSize = "moe_latent_size"
        case hybridOverridePattern = "hybrid_override_pattern"
        case layerNormEpsilon = "layer_norm_epsilon"
        case mlpBias = "mlp_bias"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_h"
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mambaNumHeads = try container.decode(Int.self, forKey: .mambaNumHeads)
        mambaHeadDim = try container.decode(Int.self, forKey: .mambaHeadDim)
        mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        ssmStateSize = try container.decode(Int.self, forKey: .ssmStateSize)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        nGroups = try container.decode(Int.self, forKey: .nGroups)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        moeLatentSize = try container.decodeIfPresent(Int.self, forKey: .moeLatentSize)
        layerNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-5
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        // Handle hybrid_override_pattern - can be string or array of strings
        if let patternString = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            hybridOverridePattern = patternString
        } else if let patternArray = try? container.decode(
            [String].self, forKey: .hybridOverridePattern)
        {
            hybridOverridePattern = patternArray.joined()
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern, in: container,
                debugDescription: "hybrid_override_pattern must be string or array of strings")
        }

        // Handle time_step_limit - can be array [min, max] or separate fields
        if let limits = try? container.decode([Float].self, forKey: .timeStepLimitMin) {
            // Actually this is time_step_limit as array
            timeStepLimitMin = limits[0]
            timeStepLimitMax = limits.count > 1 ? limits[1] : limits[0]
        } else {
            timeStepLimitMin =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin) ?? 0.0
            timeStepLimitMax =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMax)
                ?? Float.infinity
        }
    }

    /// Memberwise initializer for testing
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeSharedExpertIntermediateSize: Int,
        nRoutedExperts: Int,
        numExpertsPerTok: Int,
        hybridOverridePattern: String,
        layerNormEpsilon: Float = 1e-5,
        attentionBias: Bool = false,
        mambaProjBias: Bool = false,
        mlpBias: Bool = false,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 10000.0,
        headDim: Int? = nil,
        nSharedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1.0,
        timeStepLimitMin: Float = 0.0,
        timeStepLimitMax: Float = .infinity,
        moeLatentSize: Int? = nil
    ) {
        self.modelType = "nemotron_h"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = attentionBias
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridOverridePattern = hybridOverridePattern
        self.layerNormEpsilon = layerNormEpsilon
        self.mlpBias = mlpBias
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
        self.moeLatentSize = moeLatentSize
    }
}

// MARK: - Flash MoE (Phase 2b)
//
// Nemotron-H is unique in the matrix because:
//   1. Layers are mixed: some are attention, some are Mamba, some
//      are dense MLP, some are MoE. The `mixer` field is a `Module`
//      and the FlashMoE conformance only cares about the MoE blocks.
//   2. Expert layout is 2-proj SwitchLinear (fc1 + fc2 with relu²)
//      not 3-proj SwitchGLU. The shim must be constructed with
//      `isSwitchGLU=false` — `FlashMoE.apply` picks this up from
//      the layout enum (`.textPathSwitchMLP` → `isGLU=false`).
//   3. NemotronHLatentMoE (Nemotron-3-Super-120B) wraps the expert
//      block in a latent projection. For now we only wire FlashMoE
//      to the non-latent `NemotronHMoE` variant — the latent case
//      is rare and a follow-up.

extension NemotronHModel: FlashMoEReplaceable {
    public var flashMoELayers: [FlashMoELayer] {
        // Reach into the backbone's layers. NemotronHModel owns a
        // `backbone: NemotronHBackbone` field; we expose its layers
        // via the same pattern the LoRA conformance uses.
        backbone.layers
    }
}

extension NemotronHBlock: FlashMoELayer {
    public var flashMoELayout: FlashMoELayout {
        // Only MoE mixers get the SwitchMLP layout; attention /
        // Mamba / dense MLP blocks return `.none` so the traversal
        // skips them entirely.
        if mixer is NemotronHMoE { return .textPathSwitchMLP }
        return .none
    }

    public func replaceMoEBlock(with block: FlashMoEBlock) throws {
        guard let moe = mixer as? NemotronHMoE else { return }
        let gate = moe.gate
        block.topK = moe.numExpertsPerTok
        block.router = { x in
            let out = gate(x)
            return (indices: out.0, scores: out.1)
        }
        moe.flashMoeShim = block
    }
}
