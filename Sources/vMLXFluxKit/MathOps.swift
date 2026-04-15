import Foundation
import MLX
import MLXNN

// MARK: - Shared math building blocks
//
// Primitives used across every Flux-family model: timestep embedding
// (sinusoidal + MLP), RoPE (rotary position embedding for 2D image
// latents), attention with RoPE, SwiGLU feedforward, RMSNorm. Ported
// 1:1 from the Python mflux reference so weight names match.

// MARK: - Timestep embedding

/// Sinusoidal time embedding → MLP projection. Mirrors the pattern in
/// `mflux/models/flux/common/timesteps_projection.py` and Diffusers'
/// `TimestepEmbedding`.
public func sinusoidalTimeEmbedding(
    timesteps: MLXArray,
    embeddingDim: Int,
    maxPeriod: Float = 10000,
    scale: Float = 1000
) -> MLXArray {
    // `timesteps` is shape [B]. Return shape [B, embeddingDim].
    let half = embeddingDim / 2
    let exponent = MLXArray(
        (0..<half).map { -log(maxPeriod) * Float($0) / Float(half) }
    )
    let freqs = exp(exponent)
    let scaled = (timesteps * MLXArray(scale)).reshaped([timesteps.dim(0), 1])
    let args = scaled * freqs.reshaped([1, half])
    let sinPart = sin(args)
    let cosPart = cos(args)
    return concatenated([cosPart, sinPart], axis: -1)
}

// MARK: - RoPE for 2D latents

/// Rotary position embedding frequencies for the H×W latent grid.
/// Flux uses a concatenation of three 1D RoPE bands (time-axis stub,
/// height, width). We expose the simpler 2D variant here as a starting
/// point; the Flux-specific 3-axis version lives in the Flux1 attention
/// block where it composes with text-token axis.
/// Not `Sendable` because `MLXArray` isn't. RoPE caches live on the
/// model actor so cross-isolation passing never happens.
public struct RoPE2D {
    public let cosCache: MLXArray
    public let sinCache: MLXArray

    public init(headDim: Int, height: Int, width: Int, theta: Float = 10000) {
        let half = headDim / 2
        // Frequency bands.
        let freqs = MLXArray(
            (0..<half).map { pow(theta, -Float($0 * 2) / Float(headDim)) }
        )

        // 2D position grid, flattened to seq-len = H*W.
        var posY: [Float] = []
        var posX: [Float] = []
        for y in 0..<height {
            for x in 0..<width {
                posY.append(Float(y))
                posX.append(Float(x))
            }
        }
        let yMat = MLXArray(posY).reshaped([height * width, 1]) * freqs.reshaped([1, half])
        let xMat = MLXArray(posX).reshaped([height * width, 1]) * freqs.reshaped([1, half])

        // Interleave y and x bands so half of the dims get 2D positions.
        let yCos = cos(yMat)
        let ySin = sin(yMat)
        let xCos = cos(xMat)
        let xSin = sin(xMat)
        self.cosCache = concatenated([yCos, xCos], axis: -1)
        self.sinCache = concatenated([ySin, xSin], axis: -1)
    }

    /// Apply the rotation to a (B, H, S, D) tensor where `S` is the
    /// 2D-flattened sequence length and `D` is head dimension.
    public func apply(_ x: MLXArray) -> MLXArray {
        // Reshape cache for broadcasting: (1, 1, S, D).
        let c = cosCache.reshaped([1, 1, cosCache.dim(0), cosCache.dim(1)])
        let s = sinCache.reshaped([1, 1, sinCache.dim(0), sinCache.dim(1)])
        // Split x into two halves along the head dim, rotate.
        let d = x.dim(-1)
        let half = d / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< d]
        // Build the rotated version: (x1*cos - x2*sin, x1*sin + x2*cos)
        // Note: this is the simple variant; the full Flux RoPE also
        // handles the time axis which is concat'd in the model layer.
        let c1 = c[.ellipsis, 0 ..< half]
        let s1 = s[.ellipsis, 0 ..< half]
        let rotated1 = x1 * c1 - x2 * s1
        let rotated2 = x1 * s1 + x2 * c1
        return concatenated([rotated1, rotated2], axis: -1)
    }
}

// MARK: - RMS normalization
//
// vmlx-flux uses MLXNN's `RMSNorm(dimensions:eps:)` directly — it ships
// as `open class RMSNorm: Module, UnaryLayer` with a hardware-accelerated
// `MLXFast.rmsNorm` body. A pure-Swift fallback used to live here; it was
// removed to avoid a naming collision where my local class was shadowing
// the fast path. If you need to swap in a custom norm for a future model,
// subclass `MLXNN.RMSNorm` — don't re-declare a top-level `RMSNorm`.

// MARK: - Scaled dot-product attention with RoPE

/// Multi-head attention over a (B, S, D) sequence. Takes pre-computed
/// Q/K/V projections from the caller. `rope` is applied to Q and K
/// before the attention score is computed.
public func scaledDotProductAttention(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    rope: RoPE2D?,
    scale: Float? = nil
) -> MLXArray {
    // q/k/v are (B, H, S, D_head)
    let qRoped = rope?.apply(q) ?? q
    let kRoped = rope?.apply(k) ?? k

    let d = Float(q.dim(-1))
    let effectiveScale = scale ?? (1.0 / sqrt(d))
    // (B, H, S_q, D) @ (B, H, D, S_k) → (B, H, S_q, S_k)
    let scores = matmul(qRoped, kRoped.transposed(0, 1, 3, 2)) * MLXArray(effectiveScale)
    let attn = softmax(scores, axis: -1)
    return matmul(attn, v)  // (B, H, S_q, D)
}
