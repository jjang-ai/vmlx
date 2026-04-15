import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

// MARK: - Wan 3D Causal VAE
//
// Wan 2.x uses a 3D causal autoencoder (temporal + 2D spatial) to encode
// videos into (B, 16, T/4, H/8, W/8) latents. The decoder mirrors the
// standard AutoencoderKL structure but with Conv3d in place of Conv2d
// and causal padding along the time axis so generation is autoregressive-
// friendly.
//
// Architecture outline (decoder):
//   z (B, 16, T/4, H/8, W/8)
//     → conv_in (16 → 384, 3×3×3 causal)
//     → mid: ResBlock3D → Attn3D → ResBlock3D
//     → up_blocks[0]: 3× ResBlock3D(384) → spatial 2× upsample
//     → up_blocks[1]: 3× ResBlock3D(384→192) → spatial+temporal 2×
//     → up_blocks[2]: 3× ResBlock3D(192→96)  → spatial+temporal 2×
//     → up_blocks[3]: 3× ResBlock3D(96)
//     → norm_out + silu + conv_out (96 → 3, 3×3×3)
//   → video (B, 3, T, H, W)
//
// Status: TYPE SHAPES + FORWARD STRUCTURE. The 3D conv layers are
// stubbed via Conv2d wrappers that broadcast over the time axis, so the
// module compiles and the forward pass flows end-to-end. Real 3D
// convolutions slot in when mlx-swift exposes Conv3d (currently 2D only).

// MARK: - CausalConv3d shim

/// Stand-in for a causal 3D convolution. Currently implemented as a
/// frame-wise 2D conv followed by a 1D temporal mix. When mlx-swift
/// ships `Conv3d` we replace this with a single real call.
///
/// Weight layout follows the Wan official checkpoint: `{conv.weight,
/// conv.bias}` where weight shape is (out, in, kT, kH, kW). For now we
/// only use the kT=1 slice (2D conv) and drop the temporal kernel; this
/// is obviously wrong for real video quality but compiles and runs.
public final class CausalConv3d: Module {
    public let conv2d: Conv2d
    public let outChannels: Int

    public init(
        inChannels: Int, outChannels: Int,
        kernelSize: Int = 3, stride: Int = 1, padding: Int = 1
    ) {
        self.conv2d = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding)
        )
        self.outChannels = outChannels
        super.init()
    }

    /// Input shape: (B, C, T, H, W). Output: (B, outC, T, H, W).
    /// Collapses time into batch, runs 2D conv, uncollapses.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let h = x.dim(3)
        let w = x.dim(4)
        // (B, C, T, H, W) → (B*T, C, H, W)
        let reshaped = x.transposed(0, 2, 1, 3, 4).reshaped([b * t, c, h, w])
        let convOut = nhwcToNchw(conv2d(nchwToNhwc(reshaped)))
        // (B*T, outC, H, W) → (B, T, outC, H, W) → (B, outC, T, H, W)
        return convOut.reshaped([b, t, outChannels, h, w]).transposed(0, 2, 1, 3, 4)
    }
}

// MARK: - ResBlock3D

public final class WanResBlock3D: Module {
    public let norm1: VAEGroupNorm3D
    public let conv1: CausalConv3d
    public let norm2: VAEGroupNorm3D
    public let conv2: CausalConv3d
    public let shortcut: CausalConv3d?

    public init(inChannels: Int, outChannels: Int) {
        self.norm1 = VAEGroupNorm3D(channels: inChannels)
        self.conv1 = CausalConv3d(inChannels: inChannels, outChannels: outChannels)
        self.norm2 = VAEGroupNorm3D(channels: outChannels)
        self.conv2 = CausalConv3d(inChannels: outChannels, outChannels: outChannels)
        if inChannels != outChannels {
            self.shortcut = CausalConv3d(
                inChannels: inChannels, outChannels: outChannels,
                kernelSize: 1, stride: 1, padding: 0
            )
        } else {
            self.shortcut = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = silu(norm1(x))
        h = conv1(h)
        h = silu(norm2(h))
        h = conv2(h)
        let skip = shortcut.map { $0(x) } ?? x
        return h + skip
    }
}

// MARK: - 3D GroupNorm

/// GroupNorm for (B, C, T, H, W) tensors. Normalizes over (C_g, T, H, W)
/// per group.
public final class VAEGroupNorm3D: Module {
    public let weight: MLXArray
    public let bias: MLXArray
    public let groups: Int
    public let eps: Float

    public init(channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([channels])
        self.bias = MLXArray.zeros([channels])
        self.groups = groups
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let h = x.dim(3)
        let w = x.dim(4)
        let g = groups
        let grouped = x.reshaped([b, g, c / g, t, h, w])
        let meanVal = grouped.mean(axes: [2, 3, 4, 5], keepDims: true)
        let variance = ((grouped - meanVal) * (grouped - meanVal))
            .mean(axes: [2, 3, 4, 5], keepDims: true)
        let normed = (grouped - meanVal) * rsqrt(variance + MLXArray(eps))
        let reshaped = normed.reshaped([b, c, t, h, w])
        let w2 = weight.reshaped([1, c, 1, 1, 1])
        let b2 = bias.reshaped([1, c, 1, 1, 1])
        return reshaped * w2 + b2
    }
}

// MARK: - Upsample3D

/// 2× spatial upsample with optional 2× temporal upsample. Nearest-
/// neighbor via `repeated`.
public final class WanUpsample3D: Module {
    public let conv: CausalConv3d
    public let temporalUpsample: Bool

    public init(channels: Int, temporalUpsample: Bool) {
        self.conv = CausalConv3d(inChannels: channels, outChannels: channels)
        self.temporalUpsample = temporalUpsample
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Spatial 2× upsample on H, W.
        var up = repeated(x, count: 2, axis: 3)
        up = repeated(up, count: 2, axis: 4)
        if temporalUpsample {
            up = repeated(up, count: 2, axis: 2)
        }
        return conv(up)
    }
}

// MARK: - WanVAEDecoder

public final class WanVAEDecoder: Module {
    public let convIn: CausalConv3d
    public let midResnet1: WanResBlock3D
    public let midResnet2: WanResBlock3D
    public let upBlocks: [[WanResBlock3D]]
    public let upsamples: [WanUpsample3D?]
    public let normOut: VAEGroupNorm3D
    public let convOut: CausalConv3d

    /// Wan 2.1 VAE defaults. Channels mirror the spec: 16 latent,
    /// (96, 192, 384, 384) block progression, 3 resnets per up-block.
    public init(
        latentChannels: Int = 16,
        blockOutChannels: [Int] = [96, 192, 384, 384],
        layersPerBlock: Int = 2
    ) {
        let reversed = Array(blockOutChannels.reversed())
        let midChannels = reversed[0]

        self.convIn = CausalConv3d(
            inChannels: latentChannels, outChannels: midChannels
        )
        self.midResnet1 = WanResBlock3D(inChannels: midChannels, outChannels: midChannels)
        self.midResnet2 = WanResBlock3D(inChannels: midChannels, outChannels: midChannels)

        var blocks: [[WanResBlock3D]] = []
        var upsamples: [WanUpsample3D?] = []
        var prev = midChannels
        for (i, stageCh) in reversed.enumerated() {
            var stage: [WanResBlock3D] = []
            for layer in 0...layersPerBlock {
                let inCh = (layer == 0) ? prev : stageCh
                stage.append(WanResBlock3D(inChannels: inCh, outChannels: stageCh))
            }
            blocks.append(stage)
            prev = stageCh
            if i < reversed.count - 1 {
                // Temporal upsample only on inner stages (Wan 2.1 convention).
                let temporal = (i > 0 && i < reversed.count - 2)
                upsamples.append(WanUpsample3D(channels: stageCh, temporalUpsample: temporal))
            } else {
                upsamples.append(nil)
            }
        }
        self.upBlocks = blocks
        self.upsamples = upsamples

        self.normOut = VAEGroupNorm3D(channels: blockOutChannels[0])
        self.convOut = CausalConv3d(
            inChannels: blockOutChannels[0], outChannels: 3
        )
        super.init()
    }

    /// - Parameter latent: (B, 16, T/4, H/8, W/8) latent from Wan sampler.
    /// - Returns: (B, 3, T, H, W) video tensor in [-1, 1].
    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var h = convIn(latent)
        h = midResnet1(h)
        h = midResnet2(h)
        for (stageIdx, stage) in upBlocks.enumerated() {
            for block in stage {
                h = block(h)
            }
            if let up = upsamples[stageIdx] {
                h = up(h)
            }
        }
        h = silu(normOut(h))
        h = convOut(h)
        return h
    }

    /// Wan latent rescale (like Flux's scale/shift).
    public static let wanScaleFactor: Float = 0.2
    public static let wanShiftFactor: Float = 0.0

    public static func preprocessLatent(_ latent: MLXArray) -> MLXArray {
        return latent / MLXArray(wanScaleFactor) + MLXArray(wanShiftFactor)
    }

    /// Clamp + rescale (B, 3, T, H, W) from [-1, 1] to [0, 1] for frame-
    /// by-frame PNG/MP4 encoding.
    public static func postprocess(_ video: MLXArray) -> MLXArray {
        let clamped = clip(video, min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
        return (clamped + MLXArray(Float(1))) * MLXArray(Float(0.5))
    }
}
