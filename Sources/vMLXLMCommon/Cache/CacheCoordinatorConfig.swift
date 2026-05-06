// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation

/// Configuration for ``CacheCoordinator``, controlling which cache tiers
/// are enabled and their sizing parameters.
public struct CacheCoordinatorConfig: Sendable {

    /// Whether the in-memory paged KV cache is enabled.
    public var usePagedCache: Bool

    /// Whether the on-disk L2 cache (SQLite + safetensors) is enabled.
    public var enableDiskCache: Bool

    /// Whether the block-level on-disk L2 cache is enabled.
    ///
    /// This is separate from ``enableDiskCache``. The prompt-level disk
    /// cache stores a whole serialized prompt entry; block disk stores
    /// individual paged-prefix blocks using the same chain hashes as
    /// ``PagedCacheManager``.
    public var enableBlockDiskCache: Bool

    /// Whether the in-memory byte-budgeted L1.5 cache
    /// (`MemoryAwarePrefixCache`) is enabled. When on, the coordinator
    /// stores serialized KV arrays in RAM with memory-pressure
    /// eviction between the paged L1 and the disk L2. Gives long
    /// multi-turn sessions on constrained-RAM hardware a much larger
    /// effective working set than the fixed-block paged cache can hold.
    public var enableMemoryCache: Bool

    /// Percent of available system RAM that `MemoryAwarePrefixCache`
    /// should consume when `enableMemoryCache` is true. Passed into
    /// `MemoryCacheConfig.maxMemoryPercent`. Capped internally at 32 GB.
    public var memoryCachePercent: Double

    /// TTL for memory cache entries in minutes. 0 = no TTL.
    public var memoryCacheTTLMinutes: Double

    /// Number of tokens per paged cache block.
    public var pagedBlockSize: Int

    /// Maximum number of blocks in the paged cache pool (including sentinel).
    public var maxCacheBlocks: Int

    /// Maximum disk cache size in gigabytes.
    public var diskCacheMaxGB: Float

    /// Directory for disk cache files. If `nil`, a default temp directory is used.
    public var diskCacheDir: URL?

    /// Maximum block-level disk cache size in gigabytes.
    public var blockDiskCacheMaxGB: Double

    /// Directory for block-level disk cache files. If `nil`, a default under
    /// the disk cache directory is used.
    public var blockDiskCacheDir: URL?

    /// Maximum number of SSM state entries in the companion LRU cache.
    public var ssmMaxEntries: Int

    /// Model-specific key to prevent cross-model cache poisoning.
    /// Include model path, type, or a unique identifier. When set, cache hashes
    /// incorporate this key so different models with the same tokenizer cannot
    /// return each other's cached KV state.
    public var modelKey: String?

    public init(
        usePagedCache: Bool = true,
        enableDiskCache: Bool = true,
        enableBlockDiskCache: Bool = false,
        enableMemoryCache: Bool = false,
        memoryCachePercent: Double = 0.30,
        memoryCacheTTLMinutes: Double = 0,
        pagedBlockSize: Int = 64,
        maxCacheBlocks: Int = 1000,
        diskCacheMaxGB: Float = 10.0,
        diskCacheDir: URL? = nil,
        blockDiskCacheMaxGB: Double = 10.0,
        blockDiskCacheDir: URL? = nil,
        ssmMaxEntries: Int = 50,
        modelKey: String? = nil
    ) {
        self.usePagedCache = usePagedCache
        self.enableDiskCache = enableDiskCache
        self.enableBlockDiskCache = enableBlockDiskCache
        self.enableMemoryCache = enableMemoryCache
        self.memoryCachePercent = memoryCachePercent
        self.memoryCacheTTLMinutes = memoryCacheTTLMinutes
        self.pagedBlockSize = pagedBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.blockDiskCacheMaxGB = blockDiskCacheMaxGB
        self.blockDiskCacheDir = blockDiskCacheDir
        self.ssmMaxEntries = ssmMaxEntries
        self.modelKey = modelKey
    }
}
