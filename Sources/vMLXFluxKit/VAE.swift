import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - AutoencoderKL (Flux / Qwen / FIBO VAE)
//
// Pure-Swift port of the `diffusers.AutoencoderKL` decoder used by Flux1,
// Flux2, Qwen-Image, and FIBO. Architecture:
//
//   z (B, 16, H/8, W/8)
//     → conv_in (16 → 512, 3x3)
//     → mid_block: [ResBlock(512) → Attention(512) → ResBlock(512)]
//     → up_blocks[0]: 3× ResBlock(512) → Upsample 2×
//     → up_blocks[1]: 3× ResBlock(512→512) → Upsample 2×
//     → up_blocks[2]: 3× ResBlock(512→256) → Upsample 2×
//     → up_blocks[3]: 3× ResBlock(256→128)
//     → norm_out (GroupNorm, 32 groups)
//     → silu
//     → conv_out (128 → 3, 3x3)
//   → image (B, 3, H, W) in [-1, 1], caller rescales to [0, 1]
//
// The encoder (for img2img edit mode) is the mirror image. For now we
// only ship the decoder since every generation path needs it.
//
// Weight layout matches `diffusers` → `safetensors` naming so
// `WeightLoader` can load either a Flux checkpoint directly or a
// standalone `ae.safetensors` file. See `loadWeights(from:)` below.

// MARK: - GroupNorm

/// 32-group normalization. Diffusers uses 32 groups throughout the VAE.
public final class VAEGroupNorm: Module {
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
        // x: (B, C, H, W). Reshape to (B, groups, C/groups, H, W), normalize.
        let b = x.dim(0)
        let c = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let g = groups
        let grouped = x.reshaped([b, g, c / g, h, w])
        // Normalize across (C/g, H, W) per group.
        let meanVal = grouped.mean(axes: [2, 3, 4], keepDims: true)
        let variance = ((grouped - meanVal) * (grouped - meanVal))
            .mean(axes: [2, 3, 4], keepDims: true)
        let normalized = (grouped - meanVal) * rsqrt(variance + MLXArray(eps))
        let reshaped = normalized.reshaped([b, c, h, w])
        // Per-channel affine.
        let w2 = weight.reshaped([1, c, 1, 1])
        let b2 = bias.reshaped([1, c, 1, 1])
        return reshaped * w2 + b2
    }
}

// MARK: - ResnetBlock2D

/// Residual block used throughout the VAE: (norm → silu → conv) × 2
/// with an optional 1×1 skip conv when in_channels != out_channels.
/// Weight names match diffusers: `norm1`, `conv1`, `norm2`, `conv2`,
/// `conv_shortcut`.
public final class VAEResnetBlock: Module {
    public let norm1: VAEGroupNorm
    public let conv1: Conv2d
    public let norm2: VAEGroupNorm
    public let conv2: Conv2d
    public let convShortcut: Conv2d?

    public init(inChannels: Int, outChannels: Int) {
        self.norm1 = VAEGroupNorm(channels: inChannels)
        self.conv1 = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1
        )
        self.norm2 = VAEGroupNorm(channels: outChannels)
        self.conv2 = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1
        )
        if inChannels != outChannels {
            self.convShortcut = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: 1, stride: 1, padding: 0
            )
        } else {
            self.convShortcut = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = silu(h)
        // MLXNN Conv2d expects (B, H, W, C) layout.
        h = nhwcToNchw(conv1(nchwToNhwc(h)))
        h = norm2(h)
        h = silu(h)
        h = nhwcToNchw(conv2(nchwToNhwc(h)))
        let skip: MLXArray
        if let shortcut = convShortcut {
            skip = nhwcToNchw(shortcut(nchwToNhwc(x)))
        } else {
            skip = x
        }
        return h + skip
    }
}

// MARK: - Attention block (512ch mid-block self-attention)

/// Single-head self-attention over the flattened spatial dims of the
/// mid-block feature map. Diffusers names this `AttnBlock` in the VAE.
public final class VAEAttnBlock: Module {
    public let norm: VAEGroupNorm
    public let q: Linear
    public let k: Linear
    public let v: Linear
    public let proj: Linear
    public let channels: Int

    public init(channels: Int) {
        self.norm = VAEGroupNorm(channels: channels)
        self.q = Linear(channels, channels)
        self.k = Linear(channels, channels)
        self.v = Linear(channels, channels)
        self.proj = Linear(channels, channels)
        self.channels = channels
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, H, W). Normalize, flatten HW to sequence.
        let b = x.dim(0)
        let c = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let normed = norm(x)
        // (B, C, H, W) → (B, H*W, C)
        let seq = normed.transposed(0, 2, 3, 1).reshaped([b, h * w, c])
        let qOut = q(seq)
        let kOut = k(seq)
        let vOut = v(seq)
        // Single-head attention.
        let scale = Float(1.0 / sqrt(Float(c)))
        let scores = matmul(qOut, kOut.transposed(0, 2, 1)) * MLXArray(scale)
        let attn = softmax(scores, axis: -1)
        let out = matmul(attn, vOut)  // (B, H*W, C)
        let projected = proj(out)
        // Back to (B, C, H, W) and add residual.
        let reshaped = projected.reshaped([b, h, w, c]).transposed(0, 3, 1, 2)
        return x + reshaped
    }
}

// MARK: - Upsample

/// 2× nearest-neighbor upsample followed by a 3×3 conv. Diffusers calls
/// the conv `upsamplers.0.conv`.
public final class VAEUpsample: Module {
    public let conv: Conv2d

    public init(channels: Int) {
        self.conv = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3, stride: 1, padding: 1
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Nearest-neighbor 2× via `repeated` on H and W axes.
        let up = upsampleNearest2x(x)
        return nhwcToNchw(conv(nchwToNhwc(up)))
    }
}

/// 2× nearest upsample helper on (B, C, H, W).
public func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
    let r = repeated(x, count: 2, axis: 2)
    return repeated(r, count: 2, axis: 3)
}

// MARK: - Layout helpers
//
// MLXNN's Conv2d uses channels-last (B, H, W, C) natively. The rest of
// this file treats tensors as channels-first (B, C, H, W) because the
// diffusers weight layout + ResBlock math is clearer that way. We
// convert on every conv call.

@inlinable
public func nchwToNhwc(_ x: MLXArray) -> MLXArray {
    // (B, C, H, W) → (B, H, W, C)
    return x.transposed(0, 2, 3, 1)
}

@inlinable
public func nhwcToNchw(_ x: MLXArray) -> MLXArray {
    // (B, H, W, C) → (B, C, H, W) — reverse of nchwToNhwc.
    return x.transposed(0, 3, 1, 2)
}

// MARK: - VAEDecoder

/// Full decoder. Takes a (B, 16, H/8, W/8) latent and produces a
/// (B, 3, H, W) image in roughly [-1, 1] range. Flux uses a specific
/// scale/shift for the latent: `z = (z_raw - shift) * scale`, applied
/// before calling the decoder. Defaults are Flux's values.
public final class VAEDecoder: Module {
    public let convIn: Conv2d

    // Mid block
    public let midResnet1: VAEResnetBlock
    public let midAttn: VAEAttnBlock
    public let midResnet2: VAEResnetBlock

    // Up blocks: 4 stages, 3 resnets per stage, upsample between stages.
    // Channel progression for Flux: 512 → 512 → 512 → 256 → 128.
    public let upBlocks: [[VAEResnetBlock]]
    public let upsamples: [VAEUpsample?]

    public let normOut: VAEGroupNorm
    public let convOut: Conv2d

    /// Flux VAE shift + scale. Applied externally:
    /// `z = (latent / 0.3611) + 0.1159` for Flux1/Flux2.
    public static let fluxScaleFactor: Float = 0.3611
    public static let fluxShiftFactor: Float = 0.1159

    public init(
        latentChannels: Int = 16,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: Int = 2
    ) {
        // Diffusers iterates blockOutChannels in REVERSE for the decoder
        // (encoder goes 128→256→512→512, decoder goes 512→512→256→128).
        let reversed = Array(blockOutChannels.reversed())
        let midChannels = reversed[0]

        self.convIn = Conv2d(
            inputChannels: latentChannels,
            outputChannels: midChannels,
            kernelSize: 3, stride: 1, padding: 1
        )

        self.midResnet1 = VAEResnetBlock(inChannels: midChannels, outChannels: midChannels)
        self.midAttn    = VAEAttnBlock(channels: midChannels)
        self.midResnet2 = VAEResnetBlock(inChannels: midChannels, outChannels: midChannels)

        // Build up_blocks. Each up_block has `layersPerBlock + 1` resnets
        // in diffusers, and an upsample conv between stages (except the
        // final stage).
        var blocks: [[VAEResnetBlock]] = []
        var upsamples: [VAEUpsample?] = []
        var prevChannels = midChannels
        for (i, stageChannels) in reversed.enumerated() {
            var stageBlocks: [VAEResnetBlock] = []
            for layer in 0...layersPerBlock {
                let inCh = (layer == 0) ? prevChannels : stageChannels
                stageBlocks.append(VAEResnetBlock(
                    inChannels: inCh,
                    outChannels: stageChannels
                ))
            }
            blocks.append(stageBlocks)
            prevChannels = stageChannels
            // Upsample between every stage except the last.
            if i < reversed.count - 1 {
                upsamples.append(VAEUpsample(channels: stageChannels))
            } else {
                upsamples.append(nil)
            }
        }
        self.upBlocks = blocks
        self.upsamples = upsamples

        self.normOut = VAEGroupNorm(channels: blockOutChannels[0])
        self.convOut = Conv2d(
            inputChannels: blockOutChannels[0],
            outputChannels: 3,
            kernelSize: 3, stride: 1, padding: 1
        )
        super.init()
    }

    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        // 1. conv_in
        var h = nhwcToNchw(convIn(nchwToNhwc(latent)))

        // 2. mid block
        h = midResnet1(h)
        h = midAttn(h)
        h = midResnet2(h)

        // 3. up blocks
        for (stageIdx, stage) in upBlocks.enumerated() {
            for block in stage {
                h = block(h)
            }
            if let up = upsamples[stageIdx] {
                h = up(h)
            }
        }

        // 4. norm_out → silu → conv_out
        h = normOut(h)
        h = silu(h)
        h = nhwcToNchw(convOut(nchwToNhwc(h)))

        // Output is in ~[-1, 1]. Caller rescales to [0, 1] for PNG.
        return h
    }

    /// Apply the Flux VAE pre-decode rescale: `z = z/scale + shift`.
    public static func preprocessFluxLatent(_ latent: MLXArray) -> MLXArray {
        return latent / MLXArray(fluxScaleFactor) + MLXArray(fluxShiftFactor)
    }

    /// Clamp + rescale the decoder output from [-1, 1] to [0, 1] for PNG.
    public static func postprocess(_ image: MLXArray) -> MLXArray {
        let clamped = clip(image, min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
        return (clamped + MLXArray(Float(1))) * MLXArray(Float(0.5))
    }
}
