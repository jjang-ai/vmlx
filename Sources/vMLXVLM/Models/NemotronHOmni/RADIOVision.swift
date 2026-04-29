// RADIOVision.swift
// Native Swift/MLX port of NVIDIA's RADIO ViT (radio_v2.5-h variant) for the
// Nemotron-3-Nano-Omni-30B-A3B vision tower.
//
// Mirrors jang_tools/nemotron_omni/radio.py. Architecture:
//   • ViTPatchGenerator (CPE):
//       - Im2Patches:   (B, 3, H, W) → (B, num_patches, 3*P*P)
//       - embedder:     Linear(3*P*P → 1280, no bias)
//       - pos_embed:    bilinear interp from stored (1, 16384, 1280) max grid
//       - cls_token:    concat 10 cls/register tokens at front
//   • 32 × ViTBlock (pre-norm timm layout):
//       LayerNorm → Attention(qkv with bias, proj with bias) → residual
//       LayerNorm → MLP(fc1, GELU, fc2)  → residual
//   • NO final norm (timm sets model.norm = nn.Identity)
//
// Tensor naming on disk:
//   vision_model.radio_model.input_conditioner.norm_{mean,std}        (3,1,1)
//   vision_model.radio_model.model.patch_generator.cls_token.token    (10, 1280)
//   vision_model.radio_model.model.patch_generator.embedder.weight    (1280, 768)
//   vision_model.radio_model.model.patch_generator.pos_embed          (1, 16384, 1280)
//   vision_model.radio_model.model.patch_generator.video_embedder.weight (1280, 1536)
//   vision_model.radio_model.model.blocks.{0..31}.{norm1,norm2}.{weight,bias}  (1280,)
//   vision_model.radio_model.model.blocks.{0..31}.attn.qkv.{weight,bias}        (3840, 1280) (3840,)
//   vision_model.radio_model.model.blocks.{0..31}.attn.proj.{weight,bias}       (1280, 1280) (1280,)
//   vision_model.radio_model.model.blocks.{0..31}.mlp.fc1.{weight,bias}         (5120, 1280) (5120,)
//   vision_model.radio_model.model.blocks.{0..31}.mlp.fc2.{weight,bias}         (1280, 5120) (1280,)

import Foundation
import MLX
import MLXNN

/// Bilinear resize (1, C, H, W) → (1, C, targetH, targetW).
/// align_corners=False (Megatron / vLLM convention). Used by RADIO's CPE
/// to interpolate the stored 128×128 pos_embed down to the actual input grid.
public func nemotronOmniBilinearResize2D(
    _ x: MLXArray, targetH: Int, targetW: Int
) -> MLXArray {
    let H = x.dim(2)
    let W = x.dim(3)
    if H == targetH && W == targetW { return x }

    // Sampling grid in source coords: (i + 0.5) * source / target - 0.5
    let yFloat = (MLXArray(0 ..< Int32(targetH)).asType(.float32) + 0.5)
        * Float(H) / Float(targetH) - 0.5
    let xFloat = (MLXArray(0 ..< Int32(targetW)).asType(.float32) + 0.5)
        * Float(W) / Float(targetW) - 0.5

    let y0i = MLX.floor(yFloat).asType(.int32)
    let y1i = y0i + 1
    let x0i = MLX.floor(xFloat).asType(.int32)
    let x1i = x0i + 1
    let wy = (yFloat - y0i.asType(.float32))
    let wx = (xFloat - x0i.asType(.float32))

    let y0 = MLX.clip(y0i, min: 0, max: H - 1)
    let y1 = MLX.clip(y1i, min: 0, max: H - 1)
    let x0 = MLX.clip(x0i, min: 0, max: W - 1)
    let x1 = MLX.clip(x1i, min: 0, max: W - 1)

    // Gather: x[..., ys, :][..., xs] for each (y, x) corner combo.
    func gather(_ arr: MLXArray, _ ys: MLXArray, _ xs: MLXArray) -> MLXArray {
        let g1 = arr.take(ys, axis: 2)
        return g1.take(xs, axis: 3)
    }
    let f00 = gather(x, y0, x0)
    let f01 = gather(x, y0, x1)
    let f10 = gather(x, y1, x0)
    let f11 = gather(x, y1, x1)

    let wxB = wx.reshaped([1, 1, 1, targetW]).asType(x.dtype)
    let wyB = wy.reshaped([1, 1, targetH, 1]).asType(x.dtype)
    let one = MLXArray(1.0, dtype: x.dtype)
    return f00 * (one - wxB) * (one - wyB)
        + f01 * wxB * (one - wyB)
        + f10 * (one - wxB) * wyB
        + f11 * wxB * wyB
}

/// Pixel-shuffle helper (ps_version='v2'): (B, H, W, C) → (B, H*r, W*r, C/r²).
/// Validated against Python `pixel_shuffle` in radio.py.
public func nemotronOmniPixelShuffle(
    _ x: MLXArray, scaleFactor: Float
) -> MLXArray {
    let B = x.dim(0)
    let H = x.dim(1)
    let W = x.dim(2)
    let C = x.dim(3)
    let s = scaleFactor
    let hOut = Int(Float(H) * s)
    let wOut = Int(Float(W) * s)
    let cMid = Int(Float(C) / s)
    let cOut = Int(Float(C) / (s * s))

    // (B, H, W, C) → (B, W, H*s, C/s)
    var out = x.reshaped([B, W, hOut, cMid])
    out = out.transposed(0, 2, 1, 3) // (B, H*s, W, C/s)
    out = out.reshaped([B, hOut, wOut, cOut]) // (B, H*s, W*s, C/s²)
    out = out.transposed(0, 2, 1, 3)
    return out
}

// MARK: - ViT building blocks

public class NemotronHViTAttention: Module, UnaryLayer {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    public init(dim: Int, numHeads: Int, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qkv.wrappedValue = Linear(dim, 3 * dim, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let N = x.dim(1)
        let C = x.dim(2)
        let qkvOut = qkv(x).reshaped([B, N, 3, numHeads, headDim])
            .transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([B, N, C])
        return proj(out)
    }
}

public class NemotronHViTMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._fc1.wrappedValue = Linear(dim, hiddenDim)
        self._fc2.wrappedValue = Linear(hiddenDim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fc2(gelu(fc1(x)))
    }
}

public class NemotronHViTBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NemotronHViTAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NemotronHViTMLP

    public init(dim: Int, numHeads: Int, mlpRatio: Float) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._attn.wrappedValue = NemotronHViTAttention(dim: dim, numHeads: numHeads)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim)
        self._mlp.wrappedValue = NemotronHViTMLP(dim: dim, hiddenDim: Int(Float(dim) * mlpRatio))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attn(norm1(x))
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - Patch generator (CPE)

public class NemotronHViTPatchGenerator: Module {
    let patchSize: Int
    let embedDim: Int
    let numClsTokens: Int
    let maxGrid: Int

    @ModuleInfo(key: "embedder") var embedder: Linear
    @ModuleInfo(key: "video_embedder") var videoEmbedder: Linear
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray

    public init(
        patchSize: Int, embedDim: Int, numClsTokens: Int,
        maxGrid: Int, videoTemporalPatch: Int
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numClsTokens = numClsTokens
        self.maxGrid = maxGrid
        self._embedder.wrappedValue = Linear(3 * patchSize * patchSize, embedDim, bias: false)
        self._videoEmbedder.wrappedValue = Linear(
            videoTemporalPatch * 3 * patchSize * patchSize, embedDim, bias: false)
        self._clsToken.wrappedValue = MLXArray.zeros([numClsTokens, embedDim])
        self._posEmbed.wrappedValue = MLXArray.zeros([1, maxGrid * maxGrid, embedDim])
    }

    public func callAsFunction(_ x: MLXArray, video: Bool = false) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(2)
        let W = x.dim(3)
        let p = patchSize
        let py = H / p
        let px = W / p
        let C = x.dim(1)

        // Im2Patches: (B, C, H, W) → (B, py, px, C, p, p) → (B, py*px, C*p*p)
        var patches = x.reshaped([B, C, py, p, px, p])
        patches = patches.transposed(0, 2, 4, 1, 3, 5)
        patches = patches.reshaped([B, py * px, C * p * p])

        // Embed: video uses video_embedder (channel-stacked T frames), else embedder
        patches = video ? videoEmbedder(patches) : embedder(patches)

        // Add bilinear-interpolated pos_embed
        let pos = getPosEmbed(inputH: H, inputW: W).asType(patches.dtype)
        patches = patches + pos

        // Concat cls/register tokens at front
        let clsExpanded = clsToken.expandedDimensions(axis: 0)
        let cls = MLX.broadcast(clsExpanded, to: [B, numClsTokens, embedDim])
            .asType(patches.dtype)
        return MLX.concatenated([cls, patches], axis: 1)
    }

    private func getPosEmbed(inputH: Int, inputW: Int) -> MLXArray {
        let gy = inputH / patchSize
        let gx = inputW / patchSize
        if gy == maxGrid && gx == maxGrid {
            return posEmbed
        }
        // (1, max*max, D) → (1, max, max, D) → (1, D, max, max)
        var pe = posEmbed.reshaped([1, maxGrid, maxGrid, embedDim])
        pe = pe.transposed(0, 3, 1, 2)
        // Interpolate to max(gy, gx) square (eval-time CPE)
        let maxDim = max(gy, gx)
        pe = nemotronOmniBilinearResize2D(pe, targetH: maxDim, targetW: maxDim)
        // Window-select to (gy, gx)
        pe = pe[0..., 0..., ..<gy, ..<gx]
        // Flatten back to (1, gy*gx, D)
        pe = pe.transposed(0, 2, 3, 1).reshaped([1, gy * gx, embedDim])
        return pe
    }
}

// MARK: - Full RADIO body

public class NemotronHRADIOVisionModel: Module {
    public let embedDim: Int
    public let patchSize: Int
    public let numClsTokens: Int

    @ModuleInfo(key: "patch_generator") var patchGenerator: NemotronHViTPatchGenerator
    @ModuleInfo(key: "blocks") var blocks: [NemotronHViTBlock]

    public init(
        embedDim: Int = 1280,
        numBlocks: Int = 32,
        numHeads: Int = 16,
        mlpRatio: Float = 4.0,
        patchSize: Int = 16,
        numClsTokens: Int = 10,
        maxGrid: Int = 128,
        videoTemporalPatch: Int = 2
    ) {
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.numClsTokens = numClsTokens

        self._patchGenerator.wrappedValue = NemotronHViTPatchGenerator(
            patchSize: patchSize,
            embedDim: embedDim,
            numClsTokens: numClsTokens,
            maxGrid: maxGrid,
            videoTemporalPatch: videoTemporalPatch
        )
        self._blocks.wrappedValue = (0 ..< numBlocks).map { _ in
            NemotronHViTBlock(dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio)
        }
    }

    /// (B, 3, H, W) → (B, num_cls + num_patches, embed_dim).
    /// Caller must split off cls tokens (first numClsTokens) before
    /// applying pixel_shuffle.
    public func callAsFunction(_ x: MLXArray, video: Bool = false) -> MLXArray {
        var h = patchGenerator(x, video: video)
        for block in blocks {
            h = block(h)
        }
        return h
    }
}

/// Maps on-disk `vision_model.radio_model.*` keys to NemotronHRADIOVisionModel
/// attribute names (single-segment, no leading prefix).
public func remapRadioWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var out = [String: MLXArray]()
    let prefix = "vision_model.radio_model."
    for (k, v) in weights {
        guard k.hasPrefix(prefix) else { continue }
        let suffix = String(k.dropFirst(prefix.count))

        if suffix.hasPrefix("input_conditioner.") {
            // We don't apply input conditioner (preprocess does mean/std). Skip.
            continue
        }
        if suffix.hasPrefix("model.patch_generator.cls_token.token") {
            out["patch_generator.cls_token"] = v
        } else if suffix.hasPrefix("model.patch_generator.") {
            let inner = String(suffix.dropFirst("model.patch_generator.".count))
            out["patch_generator.\(inner)"] = v
        } else if suffix.hasPrefix("model.blocks.") {
            let inner = String(suffix.dropFirst("model.".count)) // "blocks.N.…"
            out[inner] = v
        } else if suffix == "summary_idxs" {
            // Adaptor head selection buffer — not used.
            continue
        } else if suffix.hasPrefix("model.norm.") {
            // timm sets model.norm = Identity for RADIO; skip if present.
            continue
        }
    }
    return out
}
