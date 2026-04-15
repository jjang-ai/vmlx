// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX

// MARK: - BatchCacheList

/// A CacheList wrapper that batches N per-sequence composite caches for models
/// that use multiple cache types per layer (e.g., FalconH1, BaichuanM1).
///
/// ## How It Works
///
/// Some hybrid models create a `CacheList` per layer containing sub-caches of
/// different types. For example, FalconH1 creates `CacheList(MambaCache(), KVCacheSimple())`
/// per layer — each layer has both an SSM state cache and a KV attention cache.
///
/// `BatchCacheList` wraps N such `CacheList` instances and presents batched
/// sub-caches via the subscript operator:
/// - `ArraysCache`/`MambaCache` sub-caches → wrapped as `BatchArraysCache`
/// - Standard KV sub-caches → wrapped as `BatchKVCache`
///
/// The model accesses `cache[0]` and `cache[1]` as usual — it gets the batched
/// version of each sub-cache type. After the forward pass, `splitBack()` writes
/// the updated states back to the per-sequence caches.
public final class BatchCacheList: CacheList {

    /// The original per-sequence CacheLists.
    private let slotCacheLists: [CacheList]

    /// Number of sequences in this batch.
    public let batchSize: Int

    /// Batched sub-caches, one per slot index in the CacheList.
    private var batchedSubCaches: [KVCache]

    /// Create a batched CacheList by wrapping N per-sequence CacheLists.
    ///
    /// - Parameter slotCacheLists: One CacheList per active sequence, all for
    ///   the same model layer. Must not be empty. All must have the same number
    ///   of sub-caches.
    public init(slotCacheLists: [CacheList]) {
        precondition(!slotCacheLists.isEmpty, "BatchCacheList requires at least one slot")
        self.slotCacheLists = slotCacheLists
        self.batchSize = slotCacheLists.count

        // Determine number of sub-caches by checking the first CacheList.
        // CacheList stores caches privately but we can probe via subscript.
        // We try indices 0, 1, ... until we've covered all sub-caches.
        // FalconH1 uses 2 (Mamba + KV), BaichuanM1 uses 2 (Mamba + KV).
        var batched = [KVCache]()
        for subIdx in 0 ..< 2 {  // CacheLists in practice have 2 sub-caches
            let subCaches = slotCacheLists.map { $0[subIdx] }

            if let _ = subCaches[0] as? ArraysCache {
                // SSM sub-cache — merge as BatchArraysCache
                let arrCaches = subCaches.map { $0 as! ArraysCache }
                batched.append(BatchArraysCache(slotCaches: arrCaches))
            } else {
                // KV sub-cache — merge as BatchKVCache
                batched.append(BatchKVCache(slotCaches: subCaches))
            }
        }
        self.batchedSubCaches = batched

        // Initialize CacheList with the batched sub-caches
        super.init(batched)

        // Use max offset
        self.offset = slotCacheLists.map(\.offset).max() ?? 0
    }

    /// Split batched states back to per-sequence CacheLists.
    ///
    /// Call this AFTER the model forward pass. Writes updated SSM states
    /// back to each sequence's original MambaCache sub-caches.
    public func splitBack() {
        for subCache in batchedSubCaches {
            if let batchArrays = subCache as? BatchArraysCache {
                batchArrays.splitBack()
            }
            // BatchKVCache doesn't need splitBack — the underlying slot caches
            // are already updated in-place by BatchKVCache.update()
        }
    }
}
