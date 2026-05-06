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
/// When the cache exposes an `offsetArray`, the offset is passed as an `MLXArray`
/// so the compile tracer can track it through the graph without triggering a
/// synchronous GPU readback. For all other cache types, the standard `Int`-based
/// offset path is used.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines offset), or `nil` for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let offsetArray = graphOffsetArray(for: cache) {
        return rope(x, offset: offsetArray)
    }
    return rope(x, offset: cache?.offset ?? 0)
}

/// Returns a graph-visible cache offset when the cache exposes one.
///
/// `KVCache.offset` is an `Int` API for compatibility with the upstream
/// mlx-swift model ports. Compile-safe caches keep the offset as an
/// `MLXArray` so the value can flow through `compile()` without an
/// `.item()` readback. Any model-side helper that needs positions beyond
/// RoPE (for example llama4/YARN attention scaling or shared-KV offsets)
/// should use this helper before falling back to `cache.offset`.
public func graphOffsetArray(for cache: KVCache?) -> MLXArray? {
    if let compilable = cache as? CompilableKVCache {
        return compilable.offsetArray
    }
    if let compilableTQ = cache as? CompilableTurboQuantKVCache {
        return compilableTQ.offsetArray
    }
    if let compilableRot = cache as? CompilableRotatingKVCache {
        return compilableRot.offsetArray
    }
    if let batchCache = cache as? BatchKVCache {
        return batchCache.offsetArray
    }
    return nil
}
