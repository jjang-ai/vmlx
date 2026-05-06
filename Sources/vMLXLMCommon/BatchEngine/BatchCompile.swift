// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 1A scaffolding for bucketed compile-in-batch.
//
// Full design:  `docs/superpowers/specs/2026-04-18-batch-engine-blockers-design.md`
// (specifically §4.4 cache ownership model, §6 Stage 1 bucketed compile path).
//
// This file currently ships the *types and pure-logic helpers* — bucket
// selection, cache-family classification, key equality — without any
// wiring into `BatchEngine`. Stage 1B will:
//   - Add the `BucketHandle` class that actually owns per-layer
//     `[B, H, maxLen, D]` CompilableKVCache buffers + a compiled forward
//     closure.
//   - Wire admission + prefill handoff + stepBatchDecode partitioning into
//     BatchEngine.
//   - Add integration tests that actually compile() a trace and assert
//     decode determinism vs the uncompiled path.
//
// Keeping the 1A/1B split means Stage 1A's types and their logic can be
// unit-tested in isolation. The narrow scope also means this file is safe
// to merge ahead of 1B — nothing here runs unless 1B wires it in, and
// `enableCompiledBatchDecode` defaults to `false` anyway.

import Foundation
import MLX
import MLXNN

// MARK: - CacheFamily

/// Classification of a slot's per-layer cache array into "families" that a
/// compiled trace must handle identically.
///
/// The compile path needs this because a trace is specialized by shape and
/// by the identity of the state arrays it reads/writes. Slots whose caches
/// fall into different families cannot share a trace — they have to be
/// partitioned into separate sub-groups inside `stepBatchDecode` (Stage 1B).
///
/// At Stage 1 only `.simple` is actually compile-eligible — later stages
/// add the other cases as their Compilable* variants ship.
public enum CacheFamily: Hashable, Sendable, CustomStringConvertible {
    /// All layers are `KVCacheSimple` or `CompilableKVCache`. The only family
    /// Stage 1 compiles. Standard LLM traffic.
    case simple

    /// All layers are `TurboQuantKVCache` / `CompilableTurboQuantKVCache`.
    /// Stage 2 will compile this family.
    case turboQuant

    /// All layers are `RotatingKVCache` / `CompilableRotatingKVCache` (sliding
    /// window). Stage 3.
    case rotating

    /// All layers are `MambaCache` / `ArraysCache` (hybrid SSM + dense).
    /// Stage 4.
    case mamba

    /// All layers are `CacheList` (composite — FalconH1, BaichuanM1). Stage 5.
    case cacheList

    /// Mixed cache types within one slot — currently untraceable. Falls back
    /// to the uncompiled `BatchKVCache` / `BatchArraysCache` / `BatchCacheList`
    /// path. Stage 4's `BatchArraysCache.splitBack` already handles this
    /// kind of hybrid structure at the wrapper level.
    case heterogeneous

    public var description: String {
        switch self {
        case .simple:         return "simple"
        case .turboQuant:     return "turboQuant"
        case .rotating:       return "rotating"
        case .mamba:          return "mamba"
        case .cacheList:      return "cacheList"
        case .heterogeneous:  return "heterogeneous"
        }
    }

    /// Whether this family is compile-eligible **at the current stage**.
    /// Updated as each stage ships.
    public var isCompileEligibleAtCurrentStage: Bool {
        switch self {
        case .simple:         return true   // Stage 1
        case .turboQuant:     return true   // Stage 2 (iter 21 RoPE routing fix)
        case .rotating:       return true   // Stage 3 (iter 13 wiring)
        case .mamba:          return false  // Stage 4 pending (hybrid trace grouping needed)
        case .cacheList:      return true   // Stage 5 (iter 22 wiring)
        case .heterogeneous:  return false  // never — by definition
        }
    }

    /// Classify a per-slot cache array into a family.
    ///
    /// - Parameter cache: One slot's per-layer `[KVCache]` array. Non-empty.
    /// - Returns: The family all layers agree on, or `.heterogeneous` if any
    ///   two layers disagree on family.
    ///
    /// ## How classification works
    ///
    /// For each layer we check its runtime type:
    /// - `KVCacheSimple` / `CompilableKVCache` → `.simple` member
    /// - `TurboQuantKVCache` → `.turboQuant` member
    /// - `RotatingKVCache` → `.rotating` member
    /// - `MambaCache` → `.mamba` member (subclass of ArraysCache)
    /// - `ArraysCache` → `.mamba` member (the non-Mamba ArraysCache users
    ///   like state-only caches also go here — they share split/merge semantics)
    /// - `CacheList` → `.cacheList` member
    /// - `QuantizedKVCache` → treated as `.heterogeneous` (not supported
    ///   under batch at all — see `BatchQuantize.wrapNewCacheIfNeeded`)
    /// - anything else → `.heterogeneous`
    ///
    /// If all layers agree, return that family. If not, return `.heterogeneous`.
    ///
    /// Hybrid models (e.g. Qwen3.5 Mamba: attention layers + Mamba layers)
    /// return `.heterogeneous`. That's correct — their mixed state structure
    /// is not yet compile-traceable as one trace. The Stage 4 compile path
    /// handles them via per-layer compilable variants inside a composite
    /// bucket — beyond Stage 1 scope.
    public static func classify(_ cache: [KVCache]) -> CacheFamily {
        precondition(!cache.isEmpty, "CacheFamily.classify requires non-empty cache array")

        var seen: Set<CacheFamily> = []
        for layer in cache {
            let fam = perLayerFamily(layer)
            seen.insert(fam)
            if seen.count > 1 { return .heterogeneous }
        }
        // Exactly one family across all layers.
        return seen.first ?? .heterogeneous
    }

    /// Map a single KVCache instance to its family member. Private helper for
    /// `classify` — not exposed because a single-layer family is rarely the
    /// right abstraction at call sites.
    static func perLayerFamily(_ cache: KVCache) -> CacheFamily {
        // Ordered from most-specific to least-specific to catch subclass
        // matches. `MambaCache: ArraysCache` so `MambaCache` must be checked
        // before `ArraysCache`.
        if cache is MambaCache      { return .mamba }
        if cache is ArraysCache     { return .mamba }
        if cache is CacheList       { return .cacheList }
        if cache is RotatingKVCache { return .rotating }
        if cache is TurboQuantKVCache { return .turboQuant }
        if cache is CompilableKVCache { return .simple }
        if cache is KVCacheSimple   { return .simple }
        // QuantizedKVCache (affine) and any bespoke BaseKVCache subclass:
        // not supported in compile path.
        return .heterogeneous
    }
}

// MARK: - BucketKey

/// Hash-stable identity for a compiled bucket trace.
///
/// Traces are cached in `BatchCompile.buckets` keyed by this struct. Two
/// compile-eligible slots share a trace when:
///  1. Their target bucket size is the same (pad-up-to target).
///  2. Their `maxCacheLength` matches (compile traces specialize on shape).
///  3. Their cache family matches (different state shapes → different trace).
public struct BucketKey: Hashable, Sendable, CustomStringConvertible {
    public let batchSize: Int
    public let maxCacheLength: Int
    public let family: CacheFamily

    public init(batchSize: Int, maxCacheLength: Int, family: CacheFamily) {
        precondition(batchSize >= 1, "BucketKey.batchSize must be >= 1")
        precondition(maxCacheLength >= 1, "BucketKey.maxCacheLength must be >= 1")
        self.batchSize = batchSize
        self.maxCacheLength = maxCacheLength
        self.family = family
    }

    public var description: String {
        "BucketKey(B=\(batchSize), maxLen=\(maxCacheLength), family=\(family))"
    }
}

// MARK: - BatchCompile (pure logic — Stage 1A)

/// Stateless helpers for the compile-in-batch path. The live state (trace
/// cache + bucket handles) is added in Stage 1B as an actor wrapping this
/// namespace.
///
/// Everything here is a pure function so it can be unit-tested in isolation
/// without standing up a BatchEngine + model.
public enum BatchCompile {

    // MARK: Bucket selection

    /// Pick the smallest bucket in `buckets` that is `>= activeCount`.
    ///
    /// Returns `nil` when:
    ///  - `activeCount <= 0` (nothing to run)
    ///  - `buckets` is empty (caller disabled buckets)
    ///  - no bucket is large enough (caller should route to uncompiled fallback)
    ///
    /// - Parameters:
    ///   - activeCount: Number of compile-eligible slots wanting to decode
    ///     in this step.
    ///   - buckets: Allowed bucket sizes, sorted ascending. Non-ascending
    ///     or duplicate entries are tolerated — the function sorts + dedups
    ///     internally so misconfigured params don't silently mis-route.
    public static func nextBucket(activeCount: Int, buckets: [Int]) -> Int? {
        guard activeCount > 0, !buckets.isEmpty else { return nil }
        // Defensive: sort + dedup so caller-provided configs can't mis-route.
        let sortedBuckets = Array(Set(buckets.filter { $0 >= 1 })).sorted()
        return sortedBuckets.first { $0 >= activeCount }
    }

    // MARK: Liveness mask utilities

    // MARK: - Compile a decode-step forward closure (Stage 1B.1)

    /// Build a compiled forward closure for a decode step.
    ///
    /// Wraps the MLX compile tracer so the returned closure accepts the
    /// token array and returns logits as `[MLXArray]` (single element
    /// holding `[B, L, V]`). The tracer captures each cache layer's
    /// `innerState()` — subsequent invocations mutate the captured cache
    /// in place via `_updateInternal`.
    ///
    /// ## Required preconditions
    ///
    /// - Every layer in `cacheRef` is `CompilableKVCache`. Other cache
    ///   types cannot yet be compile-traced because their state shapes
    ///   change step-to-step.
    /// - `cacheRef` has been materialised before this call — pending
    ///   tracer ops at compile time can corrupt state identity.
    /// - Tokens passed to the returned closure match the shape the trace
    ///   specialised for on first invocation (typically `[B, 1]`).
    ///
    /// ## Why this lives here and not on `TokenIterator`
    ///
    /// `TokenIterator.setupCompiledDecode` in `Evaluate.swift` inlines the
    /// same pattern for the single-sequence path. Stage 1B will call this
    /// utility from `BatchEngine` — keeping it in `BatchCompile` lets the
    /// batch path reuse the logic without pulling `TokenIterator`'s full
    /// state along. The two call sites can converge later once Stage 1B
    /// lands and the pattern is battle-tested.
    ///
    /// - Parameters:
    ///   - model: The language model to trace through.
    ///   - cacheRef: Array of `CompilableKVCache` instances, one per layer.
    ///     Captured by the returned closure; must not be empty.
    /// - Returns: A `@Sendable` closure mapping `[tokens]` → `[logits]`.
    public static func compileForward(
        model: any LanguageModel,
        cacheRef: [KVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(!cacheRef.isEmpty,
            "BatchCompile.compileForward requires at least one cache layer")
        // Accepts arrays where ALL layers belong to one of the shipped
        // compile-traceable cache families:
        //  - Stage 1: `CompilableKVCache` (pre-allocated, MLXArray offset)
        //  - Stage 2: `CompilableTurboQuantKVCache` (TQ with MLXArray offsetArray; iter 21)
        //  - Stage 3: `CompilableRotatingKVCache` (pre-allocated ring
        //    buffer, MLXArray idx+offset, MLX.where wrap)
        //
        // Mixed arrays across families — the compile trace specialises
        // on shape, so mixing families would require distinct traces.
        let allSimple = cacheRef.allSatisfy { $0 is CompilableKVCache }
        let allRotating = cacheRef.allSatisfy { $0 is CompilableRotatingKVCache }
        let allTQ = cacheRef.allSatisfy { $0 is CompilableTurboQuantKVCache }
        let allCacheList = cacheRef.allSatisfy { list in
            guard let compilable = list as? CompilableCacheList else { return false }
            return compilable.allSubCachesCompileReady
        }
        precondition(
            allSimple || allRotating || allTQ || allCacheList,
            "BatchCompile.compileForward requires every layer to be "
            + "CompilableKVCache OR CompilableRotatingKVCache OR "
            + "CompilableTurboQuantKVCache OR CompilableCacheList "
            + "(with all sub-caches compile-ready)."
        )

        let capturedModel = model
        let captured = cacheRef

        return compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured.isEmpty ? nil : captured,
                state: nil
            )
            return [result.logits]
        }
    }

    // MARK: - Stage 1B.4 scaffold (NOT YET WIRED)

    /// Stage 1B.4 placeholder — owns a per-bucket shared cache buffer
    /// and a compiled forward closure for `maxBatchSize > 1` deployments.
    ///
    /// **Status (2026-05-02):** scaffold only. `init` fatalErrors so any
    /// accidental instantiation surfaces immediately. The full wiring
    /// (cache buffer, slot↔row lifecycle, liveness-mask plumbing,
    /// per-bucket trace) is the next iteration's work — see
    /// `STAGE-1B4-DESIGN-2026-05-02.md` for the architecture.
    ///
    /// Why this exists in the codebase before its full implementation:
    /// future agents picking up Stage 1B.4 inherit a clean entry point
    /// and don't have to redesign the API surface from scratch. The
    /// `BatchEngine.maybePromoteToBucket(slot:)` hook below targets
    /// this type.
    public final class BucketHandle: @unchecked Sendable {
        public let key: BucketKey

        /// Per-layer `[B, H, maxLen, D]` cache buffers. Empty in the
        /// scaffold; the full impl populates one entry per layer at
        /// admission time.
        public var cacheLayers: [KVCache] = []

        /// Slot ID → row index in the bucket. Refreshed at admit/finish.
        public var slotRows: [String: Int] = [:]

        /// `[B]`-shaped int32 array (0 = dead, 1 = live). The compile
        /// trace captures this; Stage 1B.4 refreshes it per step via
        /// `_updateInternal`.
        public var liveMask: MLXArray?

        /// Compiled forward closure shared across all slots in the
        /// bucket. `nil` until first prefill assembles the trace.
        public var compiledForward: (@Sendable ([MLXArray]) -> [MLXArray])?

        public init(key: BucketKey) {
            self.key = key
            // Intentionally trip if anyone wires this in before the
            // full implementation lands. Removing this fatalError is
            // the next iteration's first PR.
            preconditionFailure(
                "BucketHandle is a Stage 1B.4 scaffold — see "
                + "STAGE-1B4-DESIGN-2026-05-02.md. Production code must "
                + "stay on the `maxBatchSize == 1` Stage 1B.3 path or "
                + "the uncompiled Stage 0 path until the bucket "
                + "implementation lands.")
        }

        /// Pick the lowest-index free row, or `nil` if the bucket is full.
        /// Stage 1B.4 will use this at slot admission to assign rows.
        public func firstFreeRow() -> Int? {
            guard !slotRows.isEmpty || key.batchSize > 0 else { return nil }
            let occupied = Set(slotRows.values)
            for i in 0..<key.batchSize {
                if !occupied.contains(i) { return i }
            }
            return nil
        }
    }

    /// Build an `MLXArray` liveness mask of shape `[B]`, bool.
    ///
    /// Live rows are `true`; dead (padding) rows are `false`. Passed to
    /// `CompilableKVCache.makeMask` (Stage 1B extension) to suppress
    /// attention to and from dead rows.
    ///
    /// Callers supply `liveIndices` as the row indices occupied by real
    /// slots; the remainder of `[0, bucketSize)` is considered dead.
    ///
    /// - Parameters:
    ///   - bucketSize: The compiled bucket's `B`.
    ///   - liveIndices: Sorted list of live row indices. Must be a subset of
    ///     `0..<bucketSize`.
    /// - Returns: An `MLXArray` of shape `[bucketSize]` with dtype bool.
    public static func makeLiveMask(
        bucketSize: Int, liveIndices: [Int]
    ) -> MLXArray {
        precondition(bucketSize >= 1)
        var mask = [Bool](repeating: false, count: bucketSize)
        for i in liveIndices {
            precondition(
                i >= 0 && i < bucketSize,
                "liveIndices must all be within 0..<bucketSize"
            )
            mask[i] = true
        }
        // MLXArray of Bool requires Int32 bridge: materialize as 0/1 int32
        // and cast to bool on GPU side later. At Stage 1A we just return
        // the int32 flag array — callers at Stage 1B will route through
        // the attention mask where the comparison lives on-device.
        return MLXArray(mask.map { Int32($0 ? 1 : 0) })
    }
}
