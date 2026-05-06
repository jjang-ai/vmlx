// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 5 — `CompilableCacheList`.
//
// `CacheList` is a composite cache wrapping an array of sub-caches,
// used today by FalconH1 and BaichuanM1 models. The compile path for
// these models works iff every sub-cache is compile-compatible — i.e.,
// each element is one of: `CompilableKVCache`, `CompilableRotatingKVCache`,
// or (once landed) `CompilableTurboQuantKVCache` / `CompilableMambaCache`.
//
// This subclass doesn't add new state of its own — it just overrides
// `innerState()` to flatten the sub-caches' states in stable order and
// provides a promotion helper that converts each sub-cache to its
// compile-traceable variant when possible.

import Foundation
import MLX

/// Compile-traceable specialisation of ``CacheList`` for models that
/// use composite cache structures (FalconH1, BaichuanM1).
///
/// The composite nature means correctness is inherited from the
/// sub-caches. If every sub-cache is a compile-compatible variant,
/// the composite works under compile; if any sub-cache isn't compile-
/// compatible, the composite falls back to the uncompiled path.
///
/// ## Usage
///
/// ```swift
/// // From an existing populated CacheList whose sub-caches are
/// // KVCacheSimple / RotatingKVCache:
/// let compilable = CompilableCacheList(from: cacheList)
/// // Every sub-cache is promoted to its Compilable* variant.
/// ```
///
/// ## Scope
///
/// Stage 5 ships the subclass. Wiring into `BatchEngine` is a
/// follow-up flip of `CacheFamily.cacheList.isCompileEligibleAtCurrentStage`
/// gated on real-model FalconH1/BaichuanM1 verification.
public final class CompilableCacheList: CacheList, @unchecked Sendable {

    // MARK: - Init

    /// Direct initialiser wrapping already-compile-compatible sub-caches.
    public init(compilableSubCaches: [KVCache]) {
        super.init(compilableSubCaches)
    }

    /// Promote an existing `CacheList` to a compile-traceable composite
    /// by promoting each sub-cache to its compile-compatible variant.
    ///
    /// Sub-caches that are already compile-compatible pass through
    /// unchanged. Sub-caches with no corresponding compile variant
    /// (e.g., `TurboQuantKVCache` today — Stage 2 still has residual
    /// drift) are left as-is; callers should inspect the resulting
    /// composite via `allSubCachesCompileReady` before building a
    /// compile trace.
    ///
    /// - Parameter list: Source `CacheList` — typically produced by
    ///   `model.newCache()` on FalconH1 or BaichuanM1.
    public convenience init(from list: CacheList) {
        let promoted: [KVCache] = list.caches.map { sub in
            Self.promoteSubCache(sub)
        }
        self.init(compilableSubCaches: promoted)
    }

    /// Promote a single sub-cache to its compile-compatible variant
    /// where one exists. Returns the original cache when no variant
    /// is available.
    private static func promoteSubCache(_ sub: KVCache) -> KVCache {
        if sub is CompilableKVCache { return sub }
        if sub is CompilableRotatingKVCache { return sub }
        if sub is CompilableMambaCache { return sub }

        if let simple = sub as? KVCacheSimple {
            return CompilableKVCache(from: simple, maxLength: 4096)
        }
        if let rotating = sub as? RotatingKVCache {
            return CompilableRotatingKVCache(from: rotating)
        }
        if let mamba = sub as? MambaCache {
            return CompilableMambaCache(from: mamba)
        }

        // No known conversion — return as-is. The composite will not
        // be compile-ready; caller should check before tracing.
        return sub
    }

    // MARK: - Inspection

    /// Whether every sub-cache is a compile-compatible variant. Used
    /// by the engine-level wiring to decide whether to route through
    /// the compile path or the uncompiled fallback.
    public var allSubCachesCompileReady: Bool {
        caches.allSatisfy { sub in
            sub is CompilableKVCache
                || sub is CompilableRotatingKVCache
                || sub is CompilableMambaCache
        }
    }

    // MARK: - innerState override

    /// Flatten compile-compatible sub-caches' `innerState()` in stable
    /// order. Same semantics as the parent but documented here so
    /// future contributors see this is the tracer's capture point for
    /// composite caches.
    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }
}
