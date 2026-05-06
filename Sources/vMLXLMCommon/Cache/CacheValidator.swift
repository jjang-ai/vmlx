// SPDX-License-Identifier: Apache-2.0
//
// CacheValidator — sanity checks applied to a cache entry before it is
// restored into a live model. Catches the silent-corruption class of bugs
// where a stale or mismatched entry would otherwise be replayed against
// a model whose layer count, head dimension, or quant bit-width differ
// from what the entry was captured with.
//
// Layered defense — these checks are NOT a substitute for content
// validation. The serializer's format-version tag remains the primary
// guard; this validator catches the post-deserialize mismatches that
// the format tag can't see (e.g. same v2 format but wrong layer count
// for the current model).
//
// Called from `Evaluate.generateTask` BEFORE `restoreLayerData` /
// `restoreFromDiskArrays`. Failed validation downgrades to a clean
// cache miss with a one-line warn log, so the request still completes
// — it just costs a full prefill instead of corrupting output.

import Foundation
import os

/// Per-entry validation checks. Construct from the live model context,
/// then call `validate(...)` on each candidate entry before restore.
public struct CacheValidator: Sendable {

    /// Number of attention layers in the live model. An entry with a
    /// different layer count cannot be safely restored — restore would
    /// either truncate state (fewer layers) or leave layers uninitialized
    /// (more layers).
    public let modelLayerCount: Int

    /// Optional model identity fingerprint. Two entries with the same
    /// modelKey are cross-checked structurally; entries with different
    /// modelKey are rejected outright. `nil` disables fingerprint check
    /// (test fixtures, dev mode).
    public let modelFingerprint: String?

    /// Maximum prompt token count the live model can handle. Cached
    /// entries whose offset exceeds this are rejected because the
    /// restored state would push generation past the model's context.
    public let maxPromptTokens: Int

    /// Iter 143 — per-KV-bearing-layer expected number of KV heads. When
    /// non-nil, `validateBlockKvHeads` rejects entries whose cached
    /// `n_kv_heads` doesn't match the live model. Closes the GQA-mismatch
    /// silent-corruption class: a user swapping between two quants of the
    /// same base model where the KV head dim changed (e.g. the Gemma 4
    /// sliding/full split, MLA absorb vs. dense, Qwen GQA group remap)
    /// previously got byte-compatible blocks restored into a model that
    /// expected a different KV layout — wrong-output silently.
    ///
    /// Mirror of Python `prefix_cache.py:949-1037` per-layer
    /// `num_key_value_heads` validation. `nil` keeps the historical
    /// permissive behavior for callers that don't yet know the live
    /// model's KV layout (test fixtures, legacy paths). Wire callers
    /// progressively — see the "GQA defense activation" pin in
    /// CacheValidatorContractTests for the rollout list.
    public let modelKvHeadsByLayer: [Int]?

    /// Outcome of a single validation pass. `.ok` lets the restore
    /// proceed; `.reject(reason:)` triggers a downgrade to cache miss.
    public enum Outcome: Sendable, Equatable {
        case ok
        case reject(reason: String)
    }

    public init(
        modelLayerCount: Int,
        modelFingerprint: String? = nil,
        maxPromptTokens: Int = 262_144,
        modelKvHeadsByLayer: [Int]? = nil
    ) {
        self.modelLayerCount = modelLayerCount
        self.modelFingerprint = modelFingerprint
        self.maxPromptTokens = maxPromptTokens
        self.modelKvHeadsByLayer = modelKvHeadsByLayer
    }

    /// Validate a paged-block entry before restore.
    public func validatePagedBlocks(
        blockCount: Int,
        layerCountPerBlock: Int,
        cachedTokenCount: Int
    ) -> Outcome {
        if blockCount == 0 {
            return .reject(reason: "empty block list")
        }
        if layerCountPerBlock != modelLayerCount {
            return .reject(
                reason: "layer count mismatch: cache has \(layerCountPerBlock), model has \(modelLayerCount)")
        }
        if cachedTokenCount > maxPromptTokens {
            return .reject(
                reason: "cached token count \(cachedTokenCount) exceeds model max \(maxPromptTokens)")
        }
        return .ok
    }

    /// Validate a disk-cache entry by checking the layer count keys
    /// present in the deserialized array dictionary. The serializer
    /// emits one key per layer (`__layer_kind_N__`); count them and
    /// compare against the live model.
    public func validateDiskArrays(
        keys: [String],
        cachedTokenCount: Int
    ) -> Outcome {
        let layerKindKeys = keys.filter { $0.hasPrefix("__layer_kind_") }
        if layerKindKeys.isEmpty {
            // Legacy v1 format had no per-layer markers. Trust it for
            // backward compat — the format tag check at deserialize
            // time already promoted v1 entries through a separate path.
            return .ok
        }
        if layerKindKeys.count != modelLayerCount {
            return .reject(
                reason: "disk cache layer count \(layerKindKeys.count) ≠ model \(modelLayerCount)")
        }
        if cachedTokenCount > maxPromptTokens {
            return .reject(
                reason: "disk cached tokens \(cachedTokenCount) > max \(maxPromptTokens)")
        }
        return .ok
    }

    /// Iter 143 — validate one KV-bearing layer's cached n_kv_heads
    /// against the live model. Returns `.ok` when `modelKvHeadsByLayer`
    /// is nil (caller hasn't opted in yet) so this is a safe additive
    /// check; only entries from a model with a different KV head layout
    /// are rejected.
    ///
    /// `kvLayerIndex` indexes into `modelKvHeadsByLayer` (the array of
    /// KV-BEARING layers, not the full model layer list). Caller maps
    /// from full-model index to KV layer index via the same logic used
    /// in `restoreLayerData` (KVCacheSimple / QuantizedKVCache /
    /// TurboQuantKVCache / KV-bearing CacheList sub-cache).
    public func validateBlockKvHeads(
        cachedKvHeads: Int,
        kvLayerIndex: Int
    ) -> Outcome {
        guard let expected = modelKvHeadsByLayer else { return .ok }
        guard kvLayerIndex >= 0, kvLayerIndex < expected.count else {
            return .reject(
                reason: "kv layer index \(kvLayerIndex) out of bounds (model has \(expected.count) KV-bearing layers)")
        }
        let want = expected[kvLayerIndex]
        if cachedKvHeads != want {
            return .reject(
                reason: "kv layer \(kvLayerIndex): cached n_kv_heads=\(cachedKvHeads) ≠ model \(want) — GQA mismatch (likely a quant swap on the same base model with a different KV head layout)")
        }
        return .ok
    }

    /// Iter 143 — bulk variant. Validate all KV layers' cached head
    /// counts in one call. Returns the first rejection or `.ok`.
    public func validateBlockKvHeads(cachedByLayer: [Int]) -> Outcome {
        guard let expected = modelKvHeadsByLayer else { return .ok }
        if cachedByLayer.count != expected.count {
            return .reject(
                reason: "kv layer count mismatch: cache has \(cachedByLayer.count) KV-bearing layers, model has \(expected.count)")
        }
        for (i, (got, want)) in zip(cachedByLayer, expected).enumerated() {
            if got != want {
                return .reject(
                    reason: "kv layer \(i): cached n_kv_heads=\(got) ≠ model \(want) — GQA mismatch")
            }
        }
        return .ok
    }

    /// Log a rejection at warn level so the operator can see why a
    /// cache hit was downgraded to a miss. Caller is responsible for
    /// the fallback path.
    public static func logReject(_ outcome: Outcome, tier: String) {
        guard case .reject(let reason) = outcome else { return }
        let logger = Logger(subsystem: "com.vmlx.cache", category: "validator")
        logger.warning("\(tier) cache rejected: \(reason)")
    }
}
