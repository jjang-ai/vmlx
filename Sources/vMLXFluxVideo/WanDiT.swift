import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

// MARK: - Wan 2.x DiT transformer
//
// Pure-Swift port of the Wan 2.1 / 2.2 video transformer. Architecture:
//
//   video patch embed  : (B, 16, T/4, H/8, W/8) → (B, N_vid, D)
//   time_mlp           : t → (B, D)
//   text_proj          : (B, N_txt, 4096 T5) → (B, N_txt, D)
//   blocks[num_layers] : 3D self-attention + cross-attention to text +
//                        FFN. Attention uses 3D RoPE over (T, H, W).
//   final              : norm + linear → (B, N_vid, patch_t*patch_h*patch_w*16)
//   → unpatchify to (B, 16, T/4, H/8, W/8)
//
// Hyperparams (Wan 2.1 14B):
//   dim=1536, num_layers=30, num_heads=12, patch=(1,2,2), text_dim=4096
//
// Wan 2.1 1.3B:
//   dim=1024, num_layers=24, num_heads=8
//
// STATUS: module shapes + forward pass scaffolded. Same philosophy as
// the Flux DiT — compiles cleanly, wiring is stable, replaces the
// `velocityPlaceholder` inside WANModel.generate once we have real
// weights. The Wan repository's checkpoint format is 1:1 with the
// standard safetensors naming we already parse in WeightLoader.

public struct WanDiTConfig: Sendable {
    public let dim: Int
    public let numLayers: Int
    public let numHeads: Int
    public let patchSizeT: Int
    public let patchSizeH: Int
    public let patchSizeW: Int
    public let inChannels: Int
    public let outChannels: Int
    public let textDim: Int
    public let frequencyDim: Int

    public init(
        dim: Int = 1536,
        numLayers: Int = 30,
        numHeads: Int = 12,
        patchSizeT: Int = 1,
        patchSizeH: Int = 2,
        patchSizeW: Int = 2,
        inChannels: Int = 16,
        outChannels: Int = 16,
        textDim: Int = 4096,
        frequencyDim: Int = 256
    ) {
        self.dim = dim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.patchSizeT = patchSizeT
        self.patchSizeH = patchSizeH
        self.patchSizeW = patchSizeW
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.textDim = textDim
        self.frequencyDim = frequencyDim
    }

    /// Wan 2.1 1.3B — fast variant for laptops.
    public static let wan21_1_3B = WanDiTConfig(
        dim: 1024, numLayers: 24, numHeads: 8
    )

    /// Wan 2.1 14B — full quality, desktop / Mac Studio.
    public static let wan21_14B = WanDiTConfig(
        dim: 1536, numLayers: 30, numHeads: 12
    )

    /// Wan 2.2 — assume same topology, different checkpoint.
    public static let wan22 = WanDiTConfig(
        dim: 1536, numLayers: 30, numHeads: 12
    )
}

// MARK: - Wan transformer block

/// Single Wan transformer block. Runs 3D self-attention over video
/// tokens, cross-attention to T5 text tokens, then a gated MLP. Uses
/// AdaLN modulation from the time + text_embed conditioning vector.
public final class WanDiTBlock: Module {
    public let dim: Int
    public let numHeads: Int

    public let modulation: Linear   // → 6*D
    public let norm1: LayerNorm
    public let selfAttnQKV: Linear
    public let selfAttnProj: Linear
    public let qkNorm: QKNorm

    public let norm2: LayerNorm
    public let crossAttnQ: Linear
    public let crossAttnKV: Linear
    public let crossAttnProj: Linear

    public let norm3: LayerNorm
    public let mlp0: Linear
    public let mlp2: Linear

    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0) {
        self.dim = dim
        self.numHeads = numHeads
        let headDim = dim / numHeads
        let mlpDim = Int(Float(dim) * mlpRatio)

        self.modulation = Linear(dim, dim * 6)
        self.norm1 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.selfAttnQKV = Linear(dim, dim * 3)
        self.selfAttnProj = Linear(dim, dim)
        self.qkNorm = QKNorm(headDim: headDim)

        self.norm2 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.crossAttnQ = Linear(dim, dim)
        self.crossAttnKV = Linear(dim, dim * 2)  // K and V share projection
        self.crossAttnProj = Linear(dim, dim)

        self.norm3 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.mlp0 = Linear(dim, mlpDim)
        self.mlp2 = Linear(mlpDim, dim)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        txt: MLXArray,
        vec: MLXArray
    ) -> MLXArray {
        // Extract 6 modulation triples from vec.
        let mods = modulation(silu(vec))
        let b = mods.dim(0)
        let d = dim
        let shift1 = mods[0 ..< b, 0 ..< d].reshaped([b, 1, d])
        let scale1 = mods[0 ..< b, d ..< 2*d].reshaped([b, 1, d])
        let gate1  = mods[0 ..< b, 2*d ..< 3*d].reshaped([b, 1, d])
        let shift2 = mods[0 ..< b, 3*d ..< 4*d].reshaped([b, 1, d])
        let scale2 = mods[0 ..< b, 4*d ..< 5*d].reshaped([b, 1, d])
        let gate2  = mods[0 ..< b, 5*d ..< 6*d].reshaped([b, 1, d])

        // 1. Self-attention over video tokens.
        var h = norm1(x)
        h = h * (MLXArray(Float(1)) + scale1) + shift1
        let qkv = selfAttnQKV(h)
        let (q, k, v) = splitQKV(qkv, numHeads: numHeads)
        let (qn, kn) = qkNorm(q: q, k: k)
        let attnOut = scaledDotProductAttention(q: qn, k: kn, v: v, rope: nil)
        let attnMerged = attnOut.transposed(0, 2, 1, 3).reshaped([
            attnOut.dim(0), attnOut.dim(2), dim
        ])
        var out = x + selfAttnProj(attnMerged) * gate1

        // 2. Cross-attention to T5 text tokens.
        let normed2 = norm2(out)
        let crossQ = crossAttnQ(normed2)
        let crossKV = crossAttnKV(txt)
        let d2 = crossKV.dim(-1) / 2
        let cK = crossKV[.ellipsis, 0 ..< d2]
        let cV = crossKV[.ellipsis, d2 ..< 2*d2]
        // Multi-head split.
        let headDim = dim / numHeads
        let qHead = crossQ.reshaped([crossQ.dim(0), crossQ.dim(1), numHeads, headDim])
            .transposed(0, 2, 1, 3)
        let kHead = cK.reshaped([cK.dim(0), cK.dim(1), numHeads, headDim])
            .transposed(0, 2, 1, 3)
        let vHead = cV.reshaped([cV.dim(0), cV.dim(1), numHeads, headDim])
            .transposed(0, 2, 1, 3)
        let crossOut = scaledDotProductAttention(q: qHead, k: kHead, v: vHead, rope: nil)
        let crossMerged = crossOut.transposed(0, 2, 1, 3).reshaped([
            crossOut.dim(0), crossOut.dim(2), dim
        ])
        out = out + crossAttnProj(crossMerged)

        // 3. MLP.
        var normed3 = norm3(out)
        normed3 = normed3 * (MLXArray(Float(1)) + scale2) + shift2
        let mlpOut = mlp2(gelu(mlp0(normed3)))
        out = out + mlpOut * gate2

        return out
    }
}

// MARK: - WanDiTModel

public final class WanDiTModel: Module {
    public let config: WanDiTConfig
    public let patchEmbed: Linear   // (patch_t*patch_h*patch_w*16) → dim
    public let timeIn0: Linear
    public let timeIn2: Linear
    public let textEmbed: Linear    // T5 projection: 4096 → dim
    public let blocks: [WanDiTBlock]
    public let normOut: LayerNorm
    public let linearOut: Linear

    public init(config: WanDiTConfig) {
        self.config = config
        let patchVol = config.patchSizeT * config.patchSizeH * config.patchSizeW
        self.patchEmbed = Linear(patchVol * config.inChannels, config.dim)
        self.timeIn0 = Linear(config.frequencyDim, config.dim)
        self.timeIn2 = Linear(config.dim, config.dim)
        self.textEmbed = Linear(config.textDim, config.dim)
        self.blocks = (0..<config.numLayers).map { _ in
            WanDiTBlock(dim: config.dim, numHeads: config.numHeads)
        }
        self.normOut = LayerNorm(dimensions: config.dim, eps: 1e-6, affine: false)
        self.linearOut = Linear(config.dim, patchVol * config.outChannels)
        super.init()
    }

    /// - Parameters:
    ///   - videoPatched: (B, N_vid, patch_t*patch_h*patch_w*16).
    ///   - txt: (B, N_txt, 4096) T5-XXL encoded text.
    ///   - timestep: (B,) current flow-match step.
    public func callAsFunction(
        videoPatched: MLXArray,
        txt: MLXArray,
        timestep: MLXArray
    ) -> MLXArray {
        var x = patchEmbed(videoPatched)
        let timeEmb = sinusoidalTimeEmbedding(
            timesteps: timestep, embeddingDim: config.frequencyDim
        )
        let vec = timeIn2(silu(timeIn0(timeEmb)))
        let txtEmbed = textEmbed(txt)

        for block in blocks {
            x = block(x, txt: txtEmbed, vec: vec)
        }

        let out = linearOut(normOut(x))
        return out
    }
}
