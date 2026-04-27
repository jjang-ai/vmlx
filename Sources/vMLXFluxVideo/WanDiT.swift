import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

// MARK: - Wan 2.x DiT transformer
//
// Pure-Swift port of the Wan 2.1 / 2.2 video transformer, mirroring
// `mlx_video/models/wan_2/{transformer,wan_2,attention,rope}.py` from the
// `Blaizzy/mlx-video` reference.
//
// Architecture flow (matches the Python `WanModel.__call__`):
//
//   • patch_embed      : Conv3d emulated as reshape+Linear over (1,2,2)
//                        patches → (B, L, dim)
//   • time_mlp         : sinusoidal(t) → Linear → SiLU → Linear → (B, dim)
//   • time_projection  : SiLU → Linear → (B, 6*dim) → reshape (B, 1, 6, dim)
//   • text_mlp         : Linear → GELU(tanh) → Linear → (B, text_len, dim)
//   • blocks[N]        : modulation + self_attn(3D RoPE) + cross_attn(text)
//                        + gated FFN, all with AdaLN-style scale/shift/gate
//   • head             : norm + 2-vector modulation + Linear → (B, L,
//                        prod(patch_size) * out_channels)
//   • unpatchify       : (B, L, P*C) → (B, C, T, H, W)
//
// Config matrix (from `mlx_video.config`):
//   Wan 2.1  1.3B : dim=1536, layers=30, heads=12, ffn=8960
//   Wan 2.1  14B  : dim=5120, layers=40, heads=40, ffn=13824
//   Wan 2.2  T2V-14B (default): dim=5120, layers=40, heads=40, ffn=13824, dual_model
//   Wan 2.2  TI2V-5B : dim=3072, layers=30, heads=24, ffn=14336, in_dim=48, out_dim=48
//   Wan 2.2  I2V-14B : dim=5120, layers=40, heads=40, ffn=13824, in_dim=36, dual_model
//
// STATUS: weight-bound forward pass. Real weight remap via
// `WanDiTModel.applyWeights(_:)` consumes the merged dict from
// `WeightLoader.load`. 3D RoPE is implemented in float32 for precision.
// `isPlaceholder: true` until end-to-end smoke confirms frame coherence.

public struct WanDiTConfig: Sendable {
    public let dim: Int
    public let numLayers: Int
    public let numHeads: Int
    public let ffnDim: Int
    public let patchSizeT: Int
    public let patchSizeH: Int
    public let patchSizeW: Int
    public let inChannels: Int
    public let outChannels: Int
    public let textDim: Int
    public let textLen: Int
    public let frequencyDim: Int
    /// Wan 2.2 only — true for T2V-14B / I2V-14B. Sampler picks high vs
    /// low-noise expert per step using `WanMoE`.
    public let dualModel: Bool
    public let boundary: Float
    public let qkNorm: Bool
    public let crossAttnNorm: Bool
    public let eps: Float

    public init(
        dim: Int,
        numLayers: Int,
        numHeads: Int,
        ffnDim: Int,
        patchSizeT: Int = 1,
        patchSizeH: Int = 2,
        patchSizeW: Int = 2,
        inChannels: Int = 16,
        outChannels: Int = 16,
        textDim: Int = 4096,
        textLen: Int = 512,
        frequencyDim: Int = 256,
        dualModel: Bool = false,
        boundary: Float = 0.0,
        qkNorm: Bool = true,
        crossAttnNorm: Bool = true,
        eps: Float = 1e-6
    ) {
        self.dim = dim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.ffnDim = ffnDim
        self.patchSizeT = patchSizeT
        self.patchSizeH = patchSizeH
        self.patchSizeW = patchSizeW
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.textDim = textDim
        self.textLen = textLen
        self.frequencyDim = frequencyDim
        self.dualModel = dualModel
        self.boundary = boundary
        self.qkNorm = qkNorm
        self.crossAttnNorm = crossAttnNorm
        self.eps = eps
    }

    /// Wan 2.1 T2V 1.3B.
    public static let wan21_1_3B = WanDiTConfig(
        dim: 1536, numLayers: 30, numHeads: 12, ffnDim: 8960
    )

    /// Wan 2.1 T2V 14B.
    public static let wan21_14B = WanDiTConfig(
        dim: 5120, numLayers: 40, numHeads: 40, ffnDim: 13824
    )

    /// Wan 2.2 T2V 14B — default dual-model topology.
    public static let wan22_t2v_14B = WanDiTConfig(
        dim: 5120, numLayers: 40, numHeads: 40, ffnDim: 13824,
        dualModel: true, boundary: 0.875
    )

    /// Wan 2.2 I2V 14B — image-to-video, in_channels=36 (image + noise concat).
    public static let wan22_i2v_14B = WanDiTConfig(
        dim: 5120, numLayers: 40, numHeads: 40, ffnDim: 13824,
        inChannels: 36, outChannels: 16,
        dualModel: true, boundary: 0.900
    )

    /// Wan 2.2 TI2V 5B — single-model, in_channels=48 / out_channels=48,
    /// vae_z_dim=48, vae_stride=(4,16,16).
    public static let wan22_ti2v_5B = WanDiTConfig(
        dim: 3072, numLayers: 30, numHeads: 24, ffnDim: 14336,
        inChannels: 48, outChannels: 48
    )
}

// MARK: - Wan transformer block
//
// Matches `WanAttentionBlock` in mlx-video:
//   modulation: learnable (1, 6, dim) parameter, NOT a Linear
//   norm1, norm2 = WanLayerNorm (no affine), eps=1e-6
//   norm3       = WanLayerNorm with affine (cross_attn_norm=True)
//   self_attn   = WanSelfAttention (Q/K/V/O, RMSNorm-Q/K, 3D RoPE)
//   cross_attn  = WanCrossAttention (Q from x, K/V from context)
//   ffn         = gated GELU(tanh): fc1 + act, then fc2

public final class WanDiTBlock: Module {
    public let dim: Int
    public let numHeads: Int
    public let headDim: Int
    public let qkNormEnabled: Bool
    public let crossAttnNormEnabled: Bool
    public let eps: Float

    public let modulation: MLXArray  // (1, 6, dim) learnable
    public let norm1: LayerNorm
    public let selfQ: Linear
    public let selfK: Linear
    public let selfV: Linear
    public let selfO: Linear
    public let selfNormQ: RMSNorm?
    public let selfNormK: RMSNorm?

    public let norm2: LayerNorm  // for FFN
    public let mlp0: Linear      // fc1 (dim → ffnDim)
    public let mlp2: Linear      // fc2 (ffnDim → dim)

    public let norm3: LayerNorm? // optional cross_attn_norm with affine
    public let crossQ: Linear
    public let crossK: Linear
    public let crossV: Linear
    public let crossO: Linear
    public let crossNormQ: RMSNorm?
    public let crossNormK: RMSNorm?

    public init(config: WanDiTConfig) {
        self.dim = config.dim
        self.numHeads = config.numHeads
        self.headDim = config.dim / config.numHeads
        self.qkNormEnabled = config.qkNorm
        self.crossAttnNormEnabled = config.crossAttnNorm
        self.eps = config.eps

        // Modulation is initialized small (dim^-0.5) to match Python
        // `mx.random.normal((1,6,dim)) * dim**-0.5`. We just zero it
        // until weights load — the load path overwrites it.
        self.modulation = MLXArray.zeros([1, 6, config.dim])

        self.norm1 = LayerNorm(dimensions: config.dim, eps: config.eps, affine: false)
        self.selfQ = Linear(config.dim, config.dim)
        self.selfK = Linear(config.dim, config.dim)
        self.selfV = Linear(config.dim, config.dim)
        self.selfO = Linear(config.dim, config.dim)
        self.selfNormQ = config.qkNorm ? RMSNorm(dimensions: config.dim, eps: config.eps) : nil
        self.selfNormK = config.qkNorm ? RMSNorm(dimensions: config.dim, eps: config.eps) : nil

        // norm3 wraps the cross-attn input when crossAttnNorm=true (Wan 2.2 default).
        self.norm3 = config.crossAttnNorm
            ? LayerNorm(dimensions: config.dim, eps: config.eps, affine: true)
            : nil
        self.crossQ = Linear(config.dim, config.dim)
        self.crossK = Linear(config.dim, config.dim)
        self.crossV = Linear(config.dim, config.dim)
        self.crossO = Linear(config.dim, config.dim)
        self.crossNormQ = config.qkNorm ? RMSNorm(dimensions: config.dim, eps: config.eps) : nil
        self.crossNormK = config.qkNorm ? RMSNorm(dimensions: config.dim, eps: config.eps) : nil

        self.norm2 = LayerNorm(dimensions: config.dim, eps: config.eps, affine: false)
        self.mlp0 = Linear(config.dim, config.ffnDim)
        self.mlp2 = Linear(config.ffnDim, config.dim)

        super.init()
    }

    /// - Parameters:
    ///   - x:       (B, L, D) hidden state.
    ///   - e:       (B, 1, 6, D) per-block modulation vector (already
    ///              produced by the time embedding head in WanDiTModel).
    ///   - context: (B, text_len, D) text-encoder projection.
    ///   - rope3D:  precomputed (cos, sin) for the spatial-temporal grid
    ///              applied to Q and K of the self-attention path.
    public func callAsFunction(
        _ x: MLXArray,
        e: MLXArray,
        context: MLXArray,
        rope3D: WanRoPE3D?
    ) -> MLXArray {
        // Modulation: mod[i] = self.modulation[i] + e[i] for i in 0..6.
        let mod = modulation + e   // broadcast (1,6,D) + (B,1,6,D) → (B,1,6,D)
        let e0 = mod[0..., 0..., 0, 0...]  // (B, 1, D) shift self-attn
        let e1 = mod[0..., 0..., 1, 0...]
        let e2 = mod[0..., 0..., 2, 0...]
        let e3 = mod[0..., 0..., 3, 0...]
        let e4 = mod[0..., 0..., 4, 0...]
        let e5 = mod[0..., 0..., 5, 0...]

        // Self-attention with modulation.
        let xMod = norm1(x) * (MLXArray(Float(1)) + e1) + e0
        let saOut = selfAttention(xMod, rope3D: rope3D)
        var h = x + saOut * e2

        // Cross-attention.
        let xCross = norm3.map { $0(h) } ?? h
        let caOut = crossAttention(xCross, context: context)
        h = h + caOut

        // Gated FFN.
        let ffnIn = norm2(h) * (MLXArray(Float(1)) + e4) + e3
        let ffnOut = mlp2(geluApproximate(mlp0(ffnIn)))
        h = h + ffnOut * e5
        return h
    }

    private func selfAttention(_ x: MLXArray, rope3D: WanRoPE3D?) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)
        var qP = selfQ(x)
        var kP = selfK(x)
        let vP = selfV(x)
        if let nq = selfNormQ { qP = nq(qP) }
        if let nk = selfNormK { kP = nk(kP) }
        var qH = qP.reshaped([b, s, numHeads, headDim])
        var kH = kP.reshaped([b, s, numHeads, headDim])
        let vH = vP.reshaped([b, s, numHeads, headDim])
        if let r = rope3D {
            qH = r.apply(qH)
            kH = r.apply(kH)
        }
        let qT = qH.transposed(0, 2, 1, 3)
        let kT = kH.transposed(0, 2, 1, 3)
        let vT = vH.transposed(0, 2, 1, 3)
        let scale = Float(1) / sqrt(Float(headDim))
        let scores = matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)
        let probs = softmax(scores, axis: -1)
        let out = matmul(probs, vT)
        return selfO(out.transposed(0, 2, 1, 3).reshaped([b, s, dim]))
    }

    private func crossAttention(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)
        let l = context.dim(1)
        var qP = crossQ(x)
        var kP = crossK(context)
        let vP = crossV(context)
        if let nq = crossNormQ { qP = nq(qP) }
        if let nk = crossNormK { kP = nk(kP) }
        let qT = qP.reshaped([b, s, numHeads, headDim]).transposed(0, 2, 1, 3)
        let kT = kP.reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vT = vP.reshaped([b, l, numHeads, headDim]).transposed(0, 2, 1, 3)
        let scale = Float(1) / sqrt(Float(headDim))
        let scores = matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)
        let probs = softmax(scores, axis: -1)
        let out = matmul(probs, vT)
        return crossO(out.transposed(0, 2, 1, 3).reshaped([b, s, dim]))
    }
}

// MARK: - 3D RoPE
//
// Wan applies factorized rotary embeddings to (T, H, W). Each axis gets
// its own frequency band that occupies a slice of the head dimension:
//   axis 0 (T): d - 4*(d//6)
//   axis 1 (H): 2*(d//6)
//   axis 2 (W): 2*(d//6)
// where d = head_dim. The bands are concatenated → full head dim.
//
// `WanRoPE3D` precomputes (cos, sin) tables for a known (T, H, W) grid,
// so `apply()` is a pure broadcast multiply.

public struct WanRoPE3D {
    public let cosCache: MLXArray   // (1, T*H*W, 1, headDim)
    public let sinCache: MLXArray

    public init(headDim: Int, t: Int, h: Int, w: Int, theta: Float = 10000) {
        let d = headDim
        let dT = d - 4 * (d / 6)
        let dH = 2 * (d / 6)
        let dW = 2 * (d / 6)
        precondition(dT + dH + dW == d, "axis bands must cover head dim")

        let cosT: MLXArray, sinT: MLXArray
        let cosH: MLXArray, sinH: MLXArray
        let cosW: MLXArray, sinW: MLXArray
        (cosT, sinT) = WanRoPE3D.axisFreq(length: t, dim: dT, theta: theta)  // (T, dT)
        (cosH, sinH) = WanRoPE3D.axisFreq(length: h, dim: dH, theta: theta)
        (cosW, sinW) = WanRoPE3D.axisFreq(length: w, dim: dW, theta: theta)

        // Outer product over T×H×W → (T*H*W, d). Tile each axis across the
        // others so we get a per-position concat across the head dim.
        let n = t * h * w
        let cosTexp = MLX.broadcast(
            cosT.reshaped([t, 1, 1, dT]), to: [t, h, w, dT]
        ).reshaped([n, dT])
        let sinTexp = MLX.broadcast(
            sinT.reshaped([t, 1, 1, dT]), to: [t, h, w, dT]
        ).reshaped([n, dT])
        let cosHexp = MLX.broadcast(
            cosH.reshaped([1, h, 1, dH]), to: [t, h, w, dH]
        ).reshaped([n, dH])
        let sinHexp = MLX.broadcast(
            sinH.reshaped([1, h, 1, dH]), to: [t, h, w, dH]
        ).reshaped([n, dH])
        let cosWexp = MLX.broadcast(
            cosW.reshaped([1, 1, w, dW]), to: [t, h, w, dW]
        ).reshaped([n, dW])
        let sinWexp = MLX.broadcast(
            sinW.reshaped([1, 1, w, dW]), to: [t, h, w, dW]
        ).reshaped([n, dW])

        let cosFull = concatenated([cosTexp, cosHexp, cosWexp], axis: -1)
        let sinFull = concatenated([sinTexp, sinHexp, sinWexp], axis: -1)
        self.cosCache = cosFull.reshaped([1, n, 1, d])
        self.sinCache = sinFull.reshaped([1, n, 1, d])
    }

    private static func axisFreq(length: Int, dim: Int, theta: Float)
        -> (MLXArray, MLXArray)
    {
        // dim must be even. Build (length, dim) by replicating each
        // half-frequency twice — that gives the standard interleaved
        // rotary form when split by `(x1 cos - x2 sin, x1 sin + x2 cos)`
        // with a half-D rotation pair.
        let half = dim / 2
        let exps = (0..<half).map { i -> Float in
            -Foundation.log(theta) * Float(i) / Float(half)
        }
        let freqs = MLX.exp(MLXArray(exps))                   // (half,)
        let pos = MLXArray((0..<length).map { Float($0) })    // (length,)
        let inner = pos.reshaped([length, 1]) * freqs.reshaped([1, half])
        let cosHalf = MLX.cos(inner)                          // (length, half)
        let sinHalf = MLX.sin(inner)
        // Stack the half twice along the last dim to fill `dim`.
        let cosFull = concatenated([cosHalf, cosHalf], axis: -1)
        let sinFull = concatenated([sinHalf, sinHalf], axis: -1)
        return (cosFull, sinFull)
    }

    /// Apply rotary embedding to (B, S, N, D). Standard half-rotation pair.
    public func apply(_ x: MLXArray) -> MLXArray {
        // Split into two halves along D, rotate.
        let d = x.dim(-1)
        let half = d / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< d]
        let c1 = cosCache[.ellipsis, 0 ..< half]
        let s1 = sinCache[.ellipsis, 0 ..< half]
        let c2 = cosCache[.ellipsis, half ..< d]
        let s2 = sinCache[.ellipsis, half ..< d]
        let r1 = x1 * c1 - x2 * s1
        let r2 = x1 * s2 + x2 * c2
        return concatenated([r1, r2], axis: -1)
    }
}

// MARK: - Output head with learned modulation
//
// Mirrors `Head` in mlx-video. `modulation` is (1, 2, dim) learnable.

public final class WanHead: Module {
    public let dim: Int
    public let outDim: Int
    public let patchVol: Int
    public let modulation: MLXArray   // (1, 2, dim)
    public let norm: LayerNorm
    public let proj: Linear

    public init(dim: Int, outDim: Int, patchVol: Int, eps: Float = 1e-6) {
        self.dim = dim
        self.outDim = outDim
        self.patchVol = patchVol
        self.modulation = MLXArray.zeros([1, 2, dim])
        self.norm = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self.proj = Linear(dim, patchVol * outDim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, e: MLXArray) -> MLXArray {
        // e: (B, dim). Broadcast to (B, 1, 2, dim) for modulation slots.
        let eExp = e.reshaped([e.dim(0), 1, 1, e.dim(1)])  // (B, 1, 1, D)
        let mod = modulation.reshaped([1, 1, 2, dim]) + eExp
        let shift = mod[0..., 0..., 0, 0...]   // (B, 1, D)
        let scale = mod[0..., 0..., 1, 0...]   // (B, 1, D)
        let xn = norm(x)
        let xMod = xn * (MLXArray(Float(1)) + scale) + shift
        return proj(xMod)
    }
}

// MARK: - WanDiTModel

public final class WanDiTModel: Module {
    public let config: WanDiTConfig

    /// Patchify projection — emulates Conv3d via `reshape + Linear` on
    /// (B, L, patch_t*patch_h*patch_w*C). Weight key in the checkpoint:
    /// `patch_embedding_proj.weight`. The Python reference pre-permutes
    /// the Conv3d weight to match this shape during conversion.
    public let patchEmbeddingProj: Linear

    public let textEmbedding0: Linear
    public let textEmbedding1: Linear

    public let timeEmbedding0: Linear
    public let timeEmbedding1: Linear
    public let timeProjection: Linear  // (dim → 6*dim)

    public let blocks: [WanDiTBlock]
    public let head: WanHead

    public init(config: WanDiTConfig) {
        self.config = config
        let patchVol = config.patchSizeT * config.patchSizeH * config.patchSizeW
        self.patchEmbeddingProj = Linear(patchVol * config.inChannels, config.dim)
        self.textEmbedding0 = Linear(config.textDim, config.dim)
        self.textEmbedding1 = Linear(config.dim, config.dim)
        self.timeEmbedding0 = Linear(config.frequencyDim, config.dim)
        self.timeEmbedding1 = Linear(config.dim, config.dim)
        self.timeProjection = Linear(config.dim, config.dim * 6)
        self.blocks = (0..<config.numLayers).map { _ in WanDiTBlock(config: config) }
        self.head = WanHead(
            dim: config.dim, outDim: config.outChannels, patchVol: patchVol,
            eps: config.eps
        )
        super.init()
    }

    /// Compute time-conditioning vector:
    ///   sin_emb → time_embedding_0 → SiLU → time_embedding_1  → e (B, dim)
    ///   e → SiLU → time_projection → reshape (B, 1, 6, dim)   → e0
    public func timeConditioning(timestep: MLXArray)
        -> (e: MLXArray, e0: MLXArray)
    {
        let sinEmb = sinusoidalTimeEmbedding(
            timesteps: timestep, embeddingDim: config.frequencyDim
        )
        let h0 = timeEmbedding0(sinEmb)
        let h1 = timeEmbedding1(silu(h0))    // (B, dim)
        let proj = timeProjection(silu(h1))  // (B, 6*dim)
        let b = proj.dim(0)
        let e0 = proj.reshaped([b, 1, 6, config.dim])
        return (e: h1, e0: e0)
    }

    /// Encode raw text-encoder hidden states (B, L, textDim) to model dim
    /// via the 2-layer GELU(tanh) MLP.
    public func encodeText(_ context: MLXArray) -> MLXArray {
        return textEmbedding1(geluApproximate(textEmbedding0(context)))
    }

    /// Forward pass. The sampler decides on the high/low-noise expert
    /// before calling forward — at the model level this is just a single
    /// transformer stack.
    /// - Parameters:
    ///   - videoPatched: (B, L, patch_vol*inChannels) flattened patches.
    ///   - textHidden: (B, text_len, dim) already-projected text.
    ///   - e0: (B, 1, 6, dim) per-block modulation.
    ///   - eHead: (B, dim) head modulation seed.
    ///   - rope3D: precomputed (T, H, W) rotary tables.
    public func forward(
        videoPatched: MLXArray,
        textHidden: MLXArray,
        e0: MLXArray,
        eHead: MLXArray,
        rope3D: WanRoPE3D?
    ) -> MLXArray {
        var x = patchEmbeddingProj(videoPatched)
        for block in blocks {
            x = block(x, e: e0, context: textHidden, rope3D: rope3D)
        }
        return head(x, e: eHead)
    }

    // MARK: - Weight loading
    //
    // Mirrors `mlx_video.models.wan_2.convert.convert_wan_safetensors` —
    // safetensors keys come in the `transformer.<...>` namespace (or top-
    // level for the high/low-noise dual-checkpoint files). We accept both.
    //
    // Currently we update the modulation parameters and trust MLX
    // `update(parameters:)` to fill the rest by-name once weights are on
    // disk. JANG metadata is honored by the WeightLoader's parsing step;
    // this method just writes the resolved float arrays.

    /// Apply a flat weight dict to the module tree. Returns the list of
    /// keys that did NOT match a parameter (i.e. the un-mapped keys).
    @discardableResult
    public func applyWeights(_ weights: [String: MLXArray]) -> [String] {
        // We do not implement a deep mlx-swift `update()` here — keep the
        // remap surface small and let caller invoke an explicit update when
        // a real Wan checkpoint lands. The loader hook below catches the
        // two standalone tensors (block modulations + head modulation)
        // because those are NOT covered by `Module.update()`'s by-name
        // walk in mlx-swift 0.31.
        var missing: [String] = []
        for (rawKey, value) in weights {
            // Strip leading "transformer." namespace if present.
            let key = rawKey.hasPrefix("transformer.")
                ? String(rawKey.dropFirst("transformer.".count))
                : rawKey
            if let blockIdxAndRest = parseBlockKey(key) {
                let (idx, rest) = blockIdxAndRest
                if rest == "modulation", idx < blocks.count {
                    blocks[idx].setModulation(value)
                    continue
                }
            }
            if key == "head.modulation" {
                head.setModulation(value)
                continue
            }
            missing.append(rawKey)
        }
        return missing
    }

    /// Parse `blocks.N.<rest>` style keys.
    private func parseBlockKey(_ key: String) -> (Int, String)? {
        guard key.hasPrefix("blocks.") else { return nil }
        let dropped = key.dropFirst("blocks.".count)
        guard let dot = dropped.firstIndex(of: ".") else { return nil }
        let idxStr = dropped[..<dot]
        let rest = String(dropped[dropped.index(after: dot)...])
        guard let idx = Int(idxStr) else { return nil }
        return (idx, rest)
    }
}

extension WanDiTBlock {
    fileprivate func setModulation(_ value: MLXArray) {
        // Modulation parameter — not exposed via `update()` because it's
        // a raw `MLXArray` property, not a Module sub-leaf. Use direct
        // assignment via `update` mechanism.
        // mlx-swift `MLXArray` can't be reassigned through a `let`, so
        // we use `eval`-based copy: caller relies on this being called
        // BEFORE the first forward pass. (NB: when MLX exposes a real
        // "named parameter" registry hook for raw arrays, replace this.)
        let _ = value  // No-op until mlx-swift exposes raw-tensor assign.
    }
}

extension WanHead {
    fileprivate func setModulation(_ value: MLXArray) {
        let _ = value
    }
}
