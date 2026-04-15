// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
@preconcurrency import MLX
import os

// MARK: - CacheDetail

/// Identifies which cache tier satisfied a lookup.
public enum CacheDetail: String, Sendable {
    /// The in-memory paged KV cache (L1, block-aligned).
    case paged
    /// The in-memory byte-budgeted cache (L1.5, whole-prompt entries
    /// with memory-pressure eviction via `MemoryAwarePrefixCache`).
    case memory
    /// The on-disk L2 cache.
    case disk
    /// No cache tier had a match.
    case miss
}

// MARK: - CacheFetchResult

/// The result of a unified cache lookup across all tiers.
public enum CacheFetchResult: Sendable {
    /// A cache hit with the matched prefix data.
    ///
    /// - Parameters:
    ///   - matchedTokens: Number of tokens matched from the cache.
    ///   - remainingTokens: Tokens that still need to be computed.
    ///   - detail: Which cache tier provided the hit.
    ///   - blocks: Paged cache blocks covering the matched prefix (empty for disk hits).
    ///   - ssmStates: Companion SSM states for hybrid models, if available.
    case hit(
        matchedTokens: Int,
        remainingTokens: [Int],
        detail: CacheDetail,
        blocks: [CacheBlock],
        ssmStates: [MLXArray]?,
        diskArrays: [String: MLXArray]? = nil
    )

    /// No cache tier had a match for the given tokens.
    case miss
}

// MARK: - CacheCoordinator

/// Unified cache coordinator that cascades lookups across paged (L1),
/// disk (L2), and SSM companion caches.
///
/// The coordinator implements a tiered fetch strategy:
/// 1. Try the in-memory paged cache first (fastest).
/// 2. Fall back to the on-disk cache if the paged cache misses.
/// 3. For hybrid models (with SSM layers), also fetch companion SSM state.
///
/// Thread safety for the `_isHybrid` flag is provided by `OSAllocatedUnfairLock`.
/// Individual sub-caches handle their own internal locking.
public final class CacheCoordinator: @unchecked Sendable {

    // MARK: - Properties

    /// The configuration used to create this coordinator.
    public let config: CacheCoordinatorConfig

    /// The in-memory paged KV cache, or `nil` if disabled.
    public let pagedCache: PagedCacheManager?

    /// The on-disk L2 cache, or `nil` if disabled.
    public let diskCache: DiskCache?

    /// The in-memory byte-budgeted L1.5 cache, or `nil` if disabled.
    /// Stores whole-prompt payloads (same dict-of-MLXArray format as
    /// the disk tier) with memory-pressure eviction.
    public let memoryCache: MemoryAwarePrefixCache<[String: MLXArray]>?

    /// The SSM state companion cache for hybrid models.
    public let ssmStateCache: SSMStateCache

    /// Whether the model has hybrid (attention + SSM) layers.
    private var _isHybrid: Bool = false

    /// Lock protecting `_isHybrid`.
    private let lock = OSAllocatedUnfairLock()

    // MARK: - Initialization

    /// Creates a new cache coordinator.
    ///
    /// Sub-caches are instantiated based on the configuration flags.
    ///
    /// - Parameter config: The cache configuration to use.
    public init(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        self.config = config

        if config.usePagedCache {
            self.pagedCache = PagedCacheManager(
                blockSize: config.pagedBlockSize,
                maxBlocks: config.maxCacheBlocks,
                modelKey: config.modelKey
            )
        } else {
            self.pagedCache = nil
        }

        if config.enableDiskCache {
            let dir = config.diskCacheDir
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx_disk_cache")
            self.diskCache = DiskCache(cacheDir: dir, maxSizeGB: config.diskCacheMaxGB, modelKey: config.modelKey)
        } else {
            self.diskCache = nil
        }

        if config.enableMemoryCache {
            let memCfg = MemoryCacheConfig(
                maxMemoryPercent: config.memoryCachePercent,
                ttlMinutes: config.memoryCacheTTLMinutes
            )
            self.memoryCache = MemoryAwarePrefixCache<[String: MLXArray]>(
                config: memCfg,
                modelId: config.modelKey ?? "model",
                estimateMemory: { arrays in
                    arrays.values.reduce(0) { $0 + $1.nbytes }
                }
            )
        } else {
            self.memoryCache = nil
        }

        self.ssmStateCache = SSMStateCache(maxEntries: config.ssmMaxEntries)
    }

    // MARK: - Hybrid Flag

    /// Set whether the model is hybrid (has both attention and SSM layers).
    ///
    /// When hybrid mode is active, the coordinator will also fetch/store
    /// SSM companion states alongside the KV cache data.
    ///
    /// - Parameter isHybrid: `true` for hybrid models.
    public func setHybrid(_ isHybrid: Bool) {
        lock.withLock { _isHybrid = isHybrid }
    }

    /// Whether the model is hybrid (has both attention and SSM layers).
    public var isHybrid: Bool {
        lock.withLock { _isHybrid }
    }

    // MARK: - Store helpers

    /// Decide whether SSM companion state should be dropped on store.
    ///
    /// Exposed as a pure static helper so tests can pin the decision
    /// without needing an MLX runtime. Returns `true` (skip storage)
    /// when the model is hybrid AND the request carried a chat-template
    /// generation-prompt suffix. See the call site in
    /// `storeAfterGeneration` for the full rationale.
    internal static func shouldSkipSSMStorage(
        isHybrid: Bool, genPromptLen: Int
    ) -> Bool {
        isHybrid && genPromptLen > 0
    }

    // MARK: - Fetch

    /// Perform a tiered cache lookup for the given token sequence.
    ///
    /// The lookup cascades through cache tiers in order:
    /// 1. **Paged cache** (in-memory, block-aligned prefix matching).
    /// 2. **Disk cache** (exact match on full token sequence, then with one fewer token).
    /// 3. If all tiers miss, returns `.miss`.
    ///
    /// For hybrid models, SSM companion states are fetched alongside paged cache hits.
    ///
    /// The `mediaSalt` argument is a stable fingerprint of any VLM image or
    /// video content associated with the prompt (see ``computeMediaSalt(for:)``).
    /// When non-`nil` it is mixed into every tier's hash so VLM inputs with
    /// the same text prefix but different media don't alias. Pass `nil` for
    /// text-only inputs to preserve the exact pre-existing hash.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence to look up, including any
    ///     chat-template generation-prompt suffix (e.g. `<|im_start|>assistant\n`).
    ///   - mediaSalt: Optional VLM media fingerprint; `nil` for text-only.
    ///   - genPromptLen: Number of trailing tokens that represent the
    ///     chat template's `add_generation_prompt=true` suffix. When
    ///     non-zero, those tokens are stripped from the key used for
    ///     every sub-cache lookup so that a later multi-turn request
    ///     (whose earlier turns, now part of the prefix, didn't carry a
    ///     gen-prompt suffix when stored) can still hit. Mirrors
    ///     Python's `vmlx_engine/prefix_cache.py` semantics. Pass 0
    ///     (default) for base-model / continuation requests where the
    ///     raw token sequence should match verbatim.
    /// - Returns: A ``CacheFetchResult`` describing the outcome. When a
    ///   hit is returned, `remainingTokens` is reported against the
    ///   **full** `tokens` array — so the caller can feed it into
    ///   prefill without having to re-append the gen-prompt suffix.
    public func fetch(
        tokens: [Int],
        mediaSalt: String? = nil,
        genPromptLen: Int = 0
    ) -> CacheFetchResult {
        // The hash key for every sub-cache is the prompt WITHOUT its
        // trailing generation-prompt suffix. On a hit at `matched`, the
        // caller still needs to replay the gen-prompt tokens, so we
        // reconstitute `remainingTokens` as `tokens.dropFirst(matched)`
        // (which includes the gen prompt at the tail).
        let stripLen = max(0, min(genPromptLen, tokens.count))
        let hashTokens: [Int] = stripLen > 0
            ? Array(tokens.prefix(tokens.count - stripLen))
            : tokens

        // Tier 1: Paged cache (in-memory)
        if let pagedCache,
           let result = pagedCache.fetchPrefix(tokens: hashTokens, mediaSalt: mediaSalt)
        {
            var ssmStates: [MLXArray]? = nil
            if isHybrid {
                ssmStates = ssmStateCache.fetch(
                    tokens: hashTokens,
                    boundary: result.matchedTokens,
                    mediaSalt: mediaSalt
                )
            }
            let matched = result.matchedTokens
            return .hit(
                matchedTokens: matched,
                remainingTokens: Array(tokens.dropFirst(matched)),
                detail: .paged,
                blocks: result.blocks,
                ssmStates: ssmStates,
                diskArrays: nil
            )
        }

        // Tier 1.5: Byte-budgeted whole-prompt memory cache.
        // Skipped for VLM requests (mediaSalt non-nil) — those go
        // through paged/disk which mix the salt into their hashes.
        if let memoryCache, mediaSalt == nil {
            let result = memoryCache.fetch(tokens: hashTokens)
            if let arrays = result.cache {
                let matched = hashTokens.count - result.remainingTokens.count
                let ssmStates = resolveSSMStates(
                    forTokens: hashTokens, boundary: matched, diskArrays: arrays,
                    mediaSalt: mediaSalt)
                return .hit(
                    matchedTokens: matched,
                    remainingTokens: Array(tokens.dropFirst(matched)),
                    detail: .memory,
                    blocks: [],
                    ssmStates: ssmStates,
                    diskArrays: arrays
                )
            }
        }

        // Tier 2: Disk cache (exact match on hashTokens, then one-shorter fallback)
        if let diskCache {
            if let arrays = diskCache.fetch(tokens: hashTokens, mediaSalt: mediaSalt) {
                let matched = hashTokens.count
                let ssmStates = resolveSSMStates(
                    forTokens: hashTokens, boundary: matched, diskArrays: arrays,
                    mediaSalt: mediaSalt)
                return .hit(
                    matchedTokens: matched,
                    remainingTokens: Array(tokens.dropFirst(matched)),
                    detail: .disk,
                    blocks: [],
                    ssmStates: ssmStates,
                    diskArrays: arrays
                )
            }

            if hashTokens.count > 1 {
                let shorter = Array(hashTokens.dropLast())
                if let arrays = diskCache.fetch(tokens: shorter, mediaSalt: mediaSalt) {
                    let matched = shorter.count
                    let ssmStates = resolveSSMStates(
                        forTokens: shorter, boundary: matched, diskArrays: arrays,
                        mediaSalt: mediaSalt)
                    return .hit(
                        matchedTokens: matched,
                        remainingTokens: Array(tokens.dropFirst(matched)),
                        detail: .disk,
                        blocks: [],
                        ssmStates: ssmStates,
                        diskArrays: arrays
                    )
                }
            }
        }

        // All tiers missed
        return .miss
    }

    /// Resolve SSM companion state for a disk-cache hit on a hybrid model.
    ///
    /// Tries the in-memory L1 `SSMStateCache` first (fastest, same-process
    /// re-use). Falls back to the disk-side fold: when the bundled
    /// `__ssm_count__` / `ssm_{k}` keys are present in `diskArrays`, the
    /// L2 entry was written with companion SSM state and we can rehydrate
    /// it. The L1 cache is then warmed so subsequent fetches in the same
    /// process hit memory directly.
    ///
    /// Returns `nil` for non-hybrid models (and for hybrid models where
    /// neither tier has anything to offer).
    private func resolveSSMStates(
        forTokens tokens: [Int],
        boundary: Int,
        diskArrays: [String: MLXArray],
        mediaSalt: String? = nil
    ) -> [MLXArray]? {
        guard isHybrid else { return nil }
        if let l1 = ssmStateCache.fetch(
            tokens: tokens, boundary: boundary, mediaSalt: mediaSalt)
        {
            return l1
        }
        guard let folded = TQDiskSerializer.ssmStates(from: diskArrays) else {
            return nil
        }
        // Warm L1 so the next fetch in this process is in-memory.
        ssmStateCache.store(
            ssmStates: folded, tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt)
        return folded
    }

    // MARK: - Store

    /// Store cache data after generation completes.
    ///
    /// Distributes the data to each enabled cache tier:
    /// 1. Paged cache receives the token sequence and per-block layer data.
    /// 2. Disk cache receives flattened KV arrays keyed by token hash.
    ///    When TurboQuant-compressed layers are detected in `cache`, the disk
    ///    tier stores the 26x-smaller compressed representation via
    ///    ``TQDiskSerializer`` instead of raw float16 arrays.
    /// 3. SSM companion cache receives states for hybrid models.
    ///
    /// The `perLayerData` is the full-sequence per-layer output from
    /// ``extractLayerData(from:)``. This method splits it into block-sized
    /// chunks internally before passing to the paged cache.
    ///
    /// - Parameters:
    ///   - promptTokens: The full prompt token sequence.
    ///   - perLayerData: Per-layer KV tensors covering the entire prompt sequence.
    ///     Layers without KV data (SSM layers) are `nil`.
    ///   - ssmStates: SSM layer states for hybrid models, or `nil`.
    ///   - cache: The raw per-layer KV cache array from the model. When provided
    ///     and any layer is a TurboQuant cache in compressed phase, the disk tier
    ///     stores the compressed representation. Pass `nil` (default) to use the
    ///     standard float16 disk path.
    public func storeAfterGeneration(
        promptTokens: [Int],
        perLayerData: [(keys: MLXArray, values: MLXArray)?],
        ssmStates: [MLXArray]?,
        cache: [any KVCache]? = nil,
        mediaSalt: String? = nil,
        genPromptLen: Int = 0
    ) {
        // For thinking / chat-template models, the last `genPromptLen`
        // tokens are the chat template's generation-prompt suffix.
        // Later turns replay them from scratch, so we key the cache on
        // the prompt WITHOUT that suffix and also truncate the per-layer
        // KV data at the same boundary. Mirrors Python prefix_cache.py.
        let stripLen = max(0, min(genPromptLen, promptTokens.count))
        let storeTokens: [Int] = stripLen > 0
            ? Array(promptTokens.prefix(promptTokens.count - stripLen))
            : promptTokens
        let storeTotal = storeTokens.count
        let blockSize = config.pagedBlockSize

        // Nothing to cache if the stripped prefix is empty (would alias
        // every fresh request to the zero-length bucket).
        guard storeTotal > 0 else { return }

        // vmlx #45 / MEMORY "Hybrid SSM + Thinking" mitigation:
        // post-generation SSM state from `extractSSMStates` reflects
        // the model's state at position (prompt + generated), but we
        // key the cache on `storeTokens` = prompt-only. A later fetch
        // that matches this prefix would restore SSM state from a
        // position AHEAD of the matched tokens → position mismatch →
        // garbled output (Nemotron-H / Qwen3.5-A3B / Jamba / any
        // hybrid model). Python v1.3.36 mirrors this: skip SSM
        // storage when `gen_prompt_len > 0` and the model is hybrid.
        // The paged KV tier is still populated, so attention layers
        // can still cache-hit; only the SSM companion is elided.
        //
        // True hybrid+thinking reuse needs a "prompt-only re-derive"
        // forward pass (deferred — expensive on hot path).
        let effectiveSSMStates: [MLXArray]? =
            Self.shouldSkipSSMStorage(isHybrid: isHybrid, genPromptLen: stripLen) ? nil : ssmStates

        // SLIDING-1 (2026-04-15): the previous central guard that skipped
        // disk store for any cache containing a RotatingKVCache layer is
        // gone. TQDiskSerializer v2 now emits a `.rotating` LayerKind tag
        // and persists the ring buffer (state) plus the 5-tuple metaState
        // (keep, maxSize, step, offset, idx) so sliding-window models
        // (Gemma3/3n/4 SWA, Mistral4 with maxKVSize, MiMoV2Flash, BaichuanM1
        // CacheList wrappers, Qwen3.5-VL inherited) round-trip cleanly.
        // CacheHelpers.restoreRotatingLayer reseats both halves on hit.
        // Memory L1.5 still skips by-design (it stores the raw cache
        // object, not a layer split, so wrap state is opaque); disk L2
        // is now fully wired. Paged L1 already worked.

        // Split per-layer full-sequence data into per-block chunks for the
        // paged cache, capping at `storeTotal` tokens so the cached KV
        // exactly covers the stripped prefix.
        let blockLayerData = splitLayerDataIntoBlocks(
            perLayerData, blockSize: blockSize, totalTokens: storeTotal)

        // Store in paged cache (L1 in-memory, block-indexed).
        //
        // NOTE: paged storage is safe even with RotatingKVCache
        // present because `extractLayerData` filtered out those
        // layers upstream — the block-indexed writer only sees KV-
        // bearing entries. Sliding-attention slots are re-prefilled
        // on the next turn's remaining-token pass, which is exactly
        // what their semantics want.
        if let pagedCache {
            pagedCache.storeTokenSequence(
                tokens: storeTokens, layerData: blockLayerData, mediaSalt: mediaSalt)
        }

        // Store in disk cache (L2 on-disk, unified format v2).
        // Rotating layers are now persisted via the `.rotating` kind —
        // see SLIDING-1 comment above for the v3-class behavior in v2.
        if let diskCache, let cache {
            let arrays = TQDiskSerializer.serialize(
                cache: cache,
                ssmStates: isHybrid ? effectiveSSMStates : nil
            )
            if !arrays.isEmpty {
                diskCache.store(
                    tokens: storeTokens, arrays: arrays, mediaSalt: mediaSalt)
            }
        }

        // Store in memory cache (L1.5 byte-budgeted whole-prompt).
        // Skipped for VLM (mediaSalt non-nil) since the key doesn't
        // include media state. Memory tier now uses the same TQ v2
        // serializer as disk so rotating layers round-trip cleanly via
        // the new `.rotating` LayerKind (SLIDING-1).
        if let memoryCache, mediaSalt == nil, let cache {
            let arrays = TQDiskSerializer.serialize(
                cache: cache,
                ssmStates: isHybrid ? effectiveSSMStates : nil
            )
            if !arrays.isEmpty {
                // Classify as `.user` rather than defaulting to `.assistant`:
                // stored prefix cache entries represent turn-complete state
                // that later turns can reuse via longest-prefix match. They
                // should evict SLOWER than raw assistant-only entries so
                // long chats don't kick out the shared system+user prefix
                // every few turns. Historical finding from the deep cache
                // audit 2026-04-14 — the default-`.assistant` bucketing
                // was silently starving the system-pin priority.
                memoryCache.store(
                    tokens: storeTokens,
                    payload: arrays,
                    cacheType: .user)
            }
        }

        // Store SSM companion states in the in-memory L1 (SSMStateCache)
        // for fastest same-process reuse. Disk persistence is handled by
        // TQDiskSerializer above when we have the raw cache.
        if isHybrid, let effective = effectiveSSMStates, !effective.isEmpty {
            let boundary = min(storeTotal, blockLayerData.count * blockSize)
            ssmStateCache.store(
                ssmStates: effective,
                tokens: storeTokens,
                boundary: boundary,
                mediaSalt: mediaSalt
            )
        }
    }

    /// Split full-sequence per-layer KV data into block-sized chunks.
    ///
    /// Each block spans `blockSize` tokens along the sequence dimension (axis 2
    /// for the standard `[B, H, T, D]` layout). The last block may be shorter
    /// if `totalTokens` is not a multiple of `blockSize`.
    ///
    /// Layers that are `nil` (SSM layers without KV data) are skipped in
    /// the output — only layers with actual KV data are included.
    ///
    /// - Parameters:
    ///   - layerData: Per-layer `(keys, values)` for the full sequence, from ``extractLayerData(from:)``.
    ///   - blockSize: Number of tokens per block.
    ///   - totalTokens: Total number of tokens in the sequence.
    /// - Returns: Per-block array of per-layer `(keys, values)` tuples (non-optional, nil layers filtered out).
    private func splitLayerDataIntoBlocks(
        _ layerData: [(keys: MLXArray, values: MLXArray)?],
        blockSize: Int,
        totalTokens: Int
    ) -> [[(keys: MLXArray, values: MLXArray)]] {
        guard totalTokens > 0, !layerData.isEmpty else { return [] }

        var blocks: [[(keys: MLXArray, values: MLXArray)]] = []
        var offset = 0

        while offset < totalTokens {
            let end = min(offset + blockSize, totalTokens)
            var blockData: [(keys: MLXArray, values: MLXArray)] = []

            for kv in layerData {
                guard let kv else { continue }
                // KV tensors are [B, H, T, D] — slice along axis 2 (sequence dim)
                let slicedKeys = kv.keys[.ellipsis, offset ..< end, 0...]
                let slicedValues = kv.values[.ellipsis, offset ..< end, 0...]
                blockData.append((keys: slicedKeys, values: slicedValues))
            }

            blocks.append(blockData)
            offset = end
        }

        return blocks
    }

    // MARK: - Clear

    /// Clear all cache tiers, releasing all cached data.
    public func clear() {
        pagedCache?.clear()
        diskCache?.clear()
        memoryCache?.clear()
        ssmStateCache.clear()
    }
}
