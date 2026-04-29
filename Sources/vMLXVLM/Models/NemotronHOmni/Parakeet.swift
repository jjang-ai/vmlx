// Parakeet.swift
// Native Swift/MLX port of the Parakeet Conformer encoder for Nemotron audio.
//
// 24-layer Conformer with 1024 hidden, 8 attn heads, 4096 FF intermediate,
// 9-tap depthwise conv, 8× subsampling. Mirrors
// jang_tools/nemotron_omni/parakeet.py.
//
// Tensor naming on disk:
//   sound_encoder.encoder.subsampling.layers.{0,2,3,5,6}.{weight,bias}
//   sound_encoder.encoder.subsampling.linear.{weight,bias}
//   sound_encoder.encoder.layers.{0..23}.norm_feed_forward1.{weight,bias}
//   sound_encoder.encoder.layers.{0..23}.feed_forward1.linear{1,2}.weight
//   sound_encoder.encoder.layers.{0..23}.norm_self_att.{weight,bias}
//   sound_encoder.encoder.layers.{0..23}.self_attn.{q,k,v,o,relative_k}_proj.weight
//   sound_encoder.encoder.layers.{0..23}.self_attn.bias_{u,v}
//   sound_encoder.encoder.layers.{0..23}.norm_conv.{weight,bias}
//   sound_encoder.encoder.layers.{0..23}.conv.pointwise_conv1.weight
//   sound_encoder.encoder.layers.{0..23}.conv.depthwise_conv.weight
//   sound_encoder.encoder.layers.{0..23}.conv.norm.{weight,bias,running_mean,running_var}
//   sound_encoder.encoder.layers.{0..23}.conv.pointwise_conv2.weight
//   sound_encoder.encoder.layers.{0..23}.norm_feed_forward2.{weight,bias}
//   sound_encoder.encoder.layers.{0..23}.feed_forward2.linear{1,2}.weight
//   sound_encoder.encoder.layers.{0..23}.norm_out.{weight,bias}

import Foundation
import MLX
import MLXNN

// MARK: - Subsampling

public class NemotronHParakeetSubsampling: Module, UnaryLayer {
    @ModuleInfo(key: "layers_0") var layer0: Conv2d  // 1→256, k=3, s=2
    @ModuleInfo(key: "layers_2") var layer2: Conv2d  // 256→256 k=3 s=2 depthwise
    @ModuleInfo(key: "layers_3") var layer3: Conv2d  // 256→256 k=1
    @ModuleInfo(key: "layers_5") var layer5: Conv2d  // 256→256 k=3 s=2 depthwise
    @ModuleInfo(key: "layers_6") var layer6: Conv2d  // 256→256 k=1
    @ModuleInfo(key: "linear") var linear: Linear

    public init(hidden: Int = 1024, channels: Int = 256) {
        self._layer0.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: channels,
            kernelSize: .init(3), stride: .init(2), padding: .init(1))
        self._layer2.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(3), stride: .init(2), padding: .init(1),
            groups: channels)
        self._layer3.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(1), stride: .init(1), padding: .init(0))
        self._layer5.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(3), stride: .init(2), padding: .init(1),
            groups: channels)
        self._layer6.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(1), stride: .init(1), padding: .init(0))
        self._linear.wrappedValue = Linear(channels * 16, hidden)
    }

    /// mel: (B, T, n_mels=128) → (B, T/8, hidden=1024).
    /// PyTorch reference uses T as H, n_mels as W, single channel.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // (B, T, M) → (B, T, M, 1) for MLX channels-last Conv2d
        var x = mel.expandedDimensions(axis: -1)
        x = relu(layer0(x))
        x = layer2(x)
        x = relu(layer3(x))
        x = layer5(x)
        x = relu(layer6(x))
        // x: (B, T_sub, M_sub, C). Flatten (C, M_sub) per timestep — matches
        // PyTorch order via transpose(1,2): (B, T_sub, C, M_sub) → flatten last 2.
        let B = x.dim(0)
        let TSub = x.dim(1)
        let MSub = x.dim(2)
        let C = x.dim(3)
        x = x.transposed(0, 1, 3, 2).reshaped([B, TSub, C * MSub])
        return linear(x)
    }
}

// MARK: - Macaron-style half feed-forward

public class NemotronHConformerFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(dim: Int, hidden: Int) {
        self._linear1.wrappedValue = Linear(dim, hidden, bias: false)
        self._linear2.wrappedValue = Linear(hidden, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear2(silu(linear1(x)))
    }
}

// MARK: - Relative-position attention helpers

/// Sinusoidal relative-position embeddings, mirroring
/// transformers.ParakeetEncoderRelPositionalEncoding.
/// Returns: (1, 2*seqLen-1, hiddenSize)
public func nemotronOmniBuildRelPosEmbeddings(
    seqLen: Int, hiddenSize: Int, base: Float = 10000
) -> MLXArray {
    let exps = stride(from: 0, to: hiddenSize, by: 2).map { Float($0) }
    let invFreqExp = MLXArray(exps) / Float(hiddenSize)
    let invFreq = MLXArray(1.0) / MLX.pow(MLXArray(base), invFreqExp)
    // position_ids: [seqLen-1, seqLen-2, ..., 0, -1, ..., -(seqLen-1)]
    let posList = stride(from: seqLen - 1, through: -seqLen + 1, by: -1).map { Float($0) }
    let positionIds = MLXArray(posList)
    // freqs: (2T-1, half)
    let freqs = positionIds.expandedDimensions(axis: -1)
        * invFreq.expandedDimensions(axis: 0)
    let sinV = MLX.sin(freqs)
    let cosV = MLX.cos(freqs)
    // Interleave sin/cos along last dim → (2T-1, hiddenSize)
    let stacked = MLX.stacked([sinV, cosV], axis: -1)
        .reshaped([2 * seqLen - 1, hiddenSize])
    return stacked.expandedDimensions(axis: 0) // (1, 2T-1, hiddenSize)
}

/// Transformer-XL relative-position shift ("skewing trick").
/// Input: (B, H, T, 2T-1) → Output: (B, H, T, T).
public func nemotronOmniRelShift(_ scores: MLXArray, seqLen: Int) -> MLXArray {
    let B = scores.dim(0)
    let H = scores.dim(1)
    let T = scores.dim(2)
    // Pad with one zero column on left: (B, H, T, 2T)
    let zeros = MLXArray.zeros([B, H, T, 1], dtype: scores.dtype)
    var padded = MLX.concatenated([zeros, scores], axis: -1)
    // Reshape and slice
    padded = padded.reshaped([B, H, 2 * T, T])
    padded = padded[0..., 0..., 1..., 0...] // (B, H, 2T-1, T)
    padded = padded.reshaped([B, H, T, 2 * T - 1])
    return padded[0..., 0..., 0..., ..<seqLen]
}

// MARK: - RelativeMultiHeadAttention

public class NemotronHRelativeMultiHeadAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "relative_k_proj") var relKProj: Linear
    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray

    public init(dim: Int, numHeads: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qProj.wrappedValue = Linear(dim, dim, bias: false)
        self._kProj.wrappedValue = Linear(dim, dim, bias: false)
        self._vProj.wrappedValue = Linear(dim, dim, bias: false)
        self._oProj.wrappedValue = Linear(dim, dim, bias: false)
        self._relKProj.wrappedValue = Linear(dim, dim, bias: false)
        self._biasU.wrappedValue = MLXArray.zeros([numHeads, headDim])
        self._biasV.wrappedValue = MLXArray.zeros([numHeads, headDim])
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let D = x.dim(2)
        let H = numHeads
        let Hd = headDim

        // Projections (B, H, T, Hd)
        let q = qProj(x).reshaped([B, T, H, Hd]).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped([B, T, H, Hd]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, T, H, Hd]).transposed(0, 2, 1, 3)

        // Sinusoidal rel-pos: (1, 2T-1, D) → relK projection
        let posEmb = nemotronOmniBuildRelPosEmbeddings(seqLen: T, hiddenSize: D)
            .asType(x.dtype)
        let relK = relKProj(posEmb).reshaped([1, 2 * T - 1, H, Hd])

        // Term (b)+(d): (Q + bias_v) · R^T → rel_shift
        let qWithV = q + biasV.expandedDimensions(axis: 0)
            .expandedDimensions(axis: 2)
            .asType(q.dtype)
        let relKT = relK.transposed(0, 2, 3, 1) // (1, H, Hd, 2T-1)
        var matrixBD = MLX.matmul(qWithV, relKT) // (B, H, T, 2T-1)
        matrixBD = nemotronOmniRelShift(matrixBD, seqLen: T)
        matrixBD = matrixBD * scale

        if let m = mask {
            matrixBD = matrixBD + m.asType(matrixBD.dtype)
        }

        // Term (a)+(c): (Q + bias_u) · K^T (handled via SDPA mask)
        let qWithU = q + biasU.expandedDimensions(axis: 0)
            .expandedDimensions(axis: 2)
            .asType(q.dtype)
        var out = MLXFast.scaledDotProductAttention(
            queries: qWithU, keys: k, values: v, scale: scale, mask: matrixBD)
        out = out.transposed(0, 2, 1, 3).reshaped([B, T, D])
        return oProj(out)
    }
}

// MARK: - Inference-only BatchNorm1d using stored running stats

public class NemotronHBatchNorm1d: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dim])
        self._bias.wrappedValue = MLXArray.zeros([dim])
        self._runningMean.wrappedValue = MLXArray.zeros([dim])
        self._runningVar.wrappedValue = MLXArray.ones([dim])
    }

    /// x: (B, T, D) → per-channel normalize.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = runningMean.asType(x.dtype)
        let invStd = MLX.rsqrt(runningVar + eps).asType(x.dtype)
        let w = weight.asType(x.dtype)
        let b = bias.asType(x.dtype)
        return (x - mean) * invStd * w + b
    }
}

// MARK: - Conformer Conv module

public class NemotronHConformerConvModule: Module, UnaryLayer {
    let dim: Int
    let kernelSize: Int

    @ModuleInfo(key: "pointwise_conv1") var pw1: Conv1d
    @ParameterInfo(key: "depthwise_conv_weight") var dwWeight: MLXArray
    @ModuleInfo(key: "norm") var norm: NemotronHBatchNorm1d
    @ModuleInfo(key: "pointwise_conv2") var pw2: Conv1d

    public init(dim: Int, kernelSize: Int) {
        self.dim = dim
        self.kernelSize = kernelSize
        self._pw1.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: 2 * dim, kernelSize: 1, bias: false)
        self._dwWeight.wrappedValue = MLXArray.zeros([dim, 1, kernelSize])
        self._norm.wrappedValue = NemotronHBatchNorm1d(dim: dim)
        self._pw2.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: false)
    }

    /// x: (B, T, D)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pw1(x) // (B, T, 2D)
        // GLU split + sigmoid gate
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * MLX.sigmoid(parts[1])
        // Depthwise conv (kernel=9, sym pad, per-channel)
        h = depthwise(h)
        h = norm(h)
        h = silu(h)
        return pw2(h)
    }

    private func depthwise(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let D = x.dim(2)
        let K = kernelSize
        let pad = (K - 1) / 2
        let padded = MLX.padded(x, widths: [.init((0, 0)), .init((pad, pad)), .init((0, 0))])
        // Weight: (D, 1, K) → (K, D)
        let w = dwWeight.reshaped([D, K]).transposed(1, 0).asType(x.dtype) // (K, D)
        var out = MLXArray.zeros([B, T, D], dtype: x.dtype)
        for i in 0 ..< K {
            let slice = padded[0..., i ..< (i + T), 0...]
            out = out + slice * w[i].reshaped([1, 1, D])
        }
        return out
    }
}

// MARK: - Conformer Block

public class NemotronHConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var nFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var ff1: NemotronHConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var nAttn: LayerNorm
    @ModuleInfo(key: "self_attn") var attn: NemotronHRelativeMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var nConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: NemotronHConformerConvModule
    @ModuleInfo(key: "norm_feed_forward2") var nFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var ff2: NemotronHConformerFeedForward
    @ModuleInfo(key: "norm_out") var nOut: LayerNorm

    public init(
        dim: Int = 1024, numHeads: Int = 8,
        ffHidden: Int = 4096, convKernel: Int = 9
    ) {
        self._nFF1.wrappedValue = LayerNorm(dimensions: dim)
        self._ff1.wrappedValue = NemotronHConformerFeedForward(dim: dim, hidden: ffHidden)
        self._nAttn.wrappedValue = LayerNorm(dimensions: dim)
        self._attn.wrappedValue = NemotronHRelativeMultiHeadAttention(dim: dim, numHeads: numHeads)
        self._nConv.wrappedValue = LayerNorm(dimensions: dim)
        self._conv.wrappedValue = NemotronHConformerConvModule(dim: dim, kernelSize: convKernel)
        self._nFF2.wrappedValue = LayerNorm(dimensions: dim)
        self._ff2.wrappedValue = NemotronHConformerFeedForward(dim: dim, hidden: ffHidden)
        self._nOut.wrappedValue = LayerNorm(dimensions: dim)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x + 0.5 * ff1(nFF1(x))
        h = h + attn(nAttn(h), mask: mask)
        h = h + conv(nConv(h))
        h = h + 0.5 * ff2(nFF2(h))
        return nOut(h)
    }
}

// MARK: - Full Encoder

public class NemotronHParakeetEncoder: Module {
    @ModuleInfo(key: "subsampling") var subsampling: NemotronHParakeetSubsampling
    @ModuleInfo(key: "layers") var layers: [NemotronHConformerBlock]

    public init(
        hiddenSize: Int = 1024,
        numLayers: Int = 24,
        numHeads: Int = 8,
        ffHidden: Int = 4096,
        convKernel: Int = 9
    ) {
        self._subsampling.wrappedValue = NemotronHParakeetSubsampling(hidden: hiddenSize)
        self._layers.wrappedValue = (0 ..< numLayers).map { _ in
            NemotronHConformerBlock(
                dim: hiddenSize, numHeads: numHeads,
                ffHidden: ffHidden, convKernel: convKernel)
        }
    }

    /// mel: (B, n_frames, n_mels=128) → (B, n_frames/8, hidden=1024).
    public func callAsFunction(_ mel: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = subsampling(mel)
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return x
    }
}

/// Maps on-disk `sound_encoder.encoder.*` keys to module attribute names.
/// Handles Conv2d weight transposition (OIHW → OHWI) and Conv1d (OIK → OKI).
public func remapParakeetWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var out = [String: MLXArray]()
    let prefix = "sound_encoder.encoder."
    for (k, v0) in weights {
        guard k.hasPrefix(prefix) else { continue }
        var suffix = String(k.dropFirst(prefix.count))

        // Skip the inline featurizer (we compute mel ourselves)
        if suffix.hasPrefix("feature_extractor.") { continue }
        // Drop num_batches_tracked counters
        if suffix.hasSuffix(".num_batches_tracked") { continue }

        // Subsampling: layers.{N}.* → layers_{N}.*
        if suffix.hasPrefix("subsampling.layers.") {
            let inner = String(suffix.dropFirst("subsampling.layers.".count))
            if let dot = inner.firstIndex(of: ".") {
                let n = inner[inner.startIndex ..< dot]
                let rest = inner[inner.index(after: dot)...]
                suffix = "subsampling.layers_\(n).\(rest)"
            }
        }

        // depthwise_conv.weight → depthwise_conv_weight (we hold raw param)
        if suffix.hasSuffix("conv.depthwise_conv.weight") {
            suffix = suffix.replacingOccurrences(
                of: "conv.depthwise_conv.weight",
                with: "conv.depthwise_conv_weight")
        }

        var v = v0

        // Conv2d weight: PyTorch (O, I, H, W) → MLX (O, H, W, I)
        if suffix.hasPrefix("subsampling.layers_") && suffix.hasSuffix(".weight") && v.ndim == 4 {
            v = v.transposed(0, 2, 3, 1)
        }

        // Conv1d (pointwise) weight: PyTorch (O, I, K) → MLX (O, K, I)
        if suffix.contains("pointwise_conv") && suffix.hasSuffix(".weight") && v.ndim == 3 {
            v = v.transposed(0, 2, 1)
        }

        out[suffix] = v
    }
    return out
}
