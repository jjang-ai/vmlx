// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// When the cache is a `CompilableKVCache`, the offset is passed as an `MLXArray`
/// (via `offsetArray`) so the compile tracer can track it through the graph without
/// triggering a synchronous GPU readback. For all other cache types, the standard
/// `Int`-based offset path is used.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines offset), or `nil` for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let compilable = cache as? CompilableKVCache {
        return rope(x, offset: compilable.offsetArray)
    }
    // Batched decode: use per-sequence [B]-shaped offsets for correct positional encoding.
    if let batchCache = cache as? BatchKVCache {
        return rope(x, offset: batchCache.offsetArray)
    }
    return rope(x, offset: cache?.offset ?? 0)
}
