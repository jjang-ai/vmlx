import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
/// Iter 143 — MLA-specific SDPA helper for DeepSeek V2/V3/V3.2/Kimi
/// K2.6 / Mistral 4 / GLM-5.1. Differs from `attentionWithCacheUpdate`:
///
/// 1. **Caller has ALREADY called `cache.update(...)`** and is passing
///    the full post-update keys/values. This helper does NOT call
///    `cache.update` again — that's how MLA's external-update flow
///    diverges from standard attention layers.
///
/// 2. **fp32 promote on L==1.** Python's deepseek_v3.py and mistral4.py
///    cast q_nope/k/v/mask to fp32 before SDPA on the L==1 absorb
///    path (decode step) because bf16 drifts ~7.0 logit magnitude on
///    MLA's compressed latent representation, leading to repetition
///    loops after ~14 generated tokens. The cast is L==1-only — bulk
///    prefill (L>1) stays in bf16/fp16 to preserve perf.
///
/// 3. **Mask is converted to ScaledDotProductAttentionMaskMode.array
///    + casted** when fp32 promotion fires (.none mask stays as-is).
///
/// Result is cast back to the source dtype before return so caller
/// state is unchanged.
public func mlaScaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    let originalDtype = queries.dtype
    // L is the sequence dimension for queries — axis 2 in [B,H,L,D].
    let L = queries.dim(2)
    let isLOne = (L == 1)
    if isLOne && originalDtype != .float32 {
        // fp32 promote — see header comment for rationale.
        let qFp32 = queries.asType(.float32)
        let kFp32 = keys.asType(.float32)
        let vFp32 = values.asType(.float32)
        let maskFp32: MLXFast.ScaledDotProductAttentionMaskMode = {
            switch mask {
            case .array(let m):
                return .array(m.asType(.float32))
            default:
                return mask
            }
        }()
        let outFp32 = MLXFast.scaledDotProductAttention(
            queries: qFp32,
            keys: kFp32,
            values: vFp32,
            scale: scale,
            mask: maskFp32
        )
        return outFp32.asType(originalDtype)
    }
    // L>1 (prefill) — stay in source dtype, no cast cost.
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask
    )
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)

        // Fixed-buffer compiled caches (CompilableKVCache and
        // CompilableRotatingKVCache) return their full static backing store
        // from update(). If the model passed `.none`, give the cache one
        // chance to supply an array mask so uninitialized tail slots do not
        // dilute attention. Ordinary caches return `.none` here for single-
        // token decode, preserving the upstream path.
        var effectiveMask = mask
        if case .none = mask {
            let cacheMask = cache.makeMask(
                n: queries.dim(2), windowSize: nil, returnArray: true)
            if case .array = cacheMask {
                effectiveMask = cacheMask
            }
        }

        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: effectiveMask
        )
    }
}
