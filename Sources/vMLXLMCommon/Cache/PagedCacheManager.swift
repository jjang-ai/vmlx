// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
import MLX
import os

// MARK: - CacheStats

/// Statistics for cache utilisation and hit/miss tracking.
public struct CacheStats: Sendable {
    public var totalBlocks: Int = 0
    public var allocatedBlocks: Int = 0
    public var freeBlocks: Int = 0
    public var cacheHits: Int = 0
    public var cacheMisses: Int = 0
    public var evictions: Int = 0
}

// MARK: - PrefixFetchResult

/// The result of a prefix-match query against the paged cache.
public struct PrefixFetchResult: Sendable {
    /// Number of tokens that were matched from the cache.
    public let matchedTokens: Int
    /// Tokens that still need to be computed.
    public let remainingTokens: [Int]
    /// Cache blocks covering the matched prefix, in order.
    public let blocks: [CacheBlock]
}

// MARK: - PagedCacheManager

/// Manages a fixed-size pool of ``CacheBlock`` instances with hash-based
/// prefix matching for KV cache reuse.
///
/// Thread safety is provided by an `OSAllocatedUnfairLock`. All public
/// methods acquire the lock before accessing shared state.
public final class PagedCacheManager: @unchecked Sendable {

    // MARK: - Properties

    /// Number of tokens per block.
    public let blockSize: Int

    /// Total number of blocks in the pool (including the null sentinel at index 0).
    public let maxBlocks: Int

    /// Model key for cache isolation (prevents cross-model hash collisions).
    public let modelKey: String?

    /// Lock for thread safety.
    private let lock = OSAllocatedUnfairLock()

    /// Pre-allocated block pool. Index 0 is the null sentinel.
    private var blocks: [CacheBlock]

    /// Free block queue (LRU eviction order).
    private let freeQueue = FreeBlockQueue()

    /// Hash-to-block mapping for prefix lookup.
    private let hashMap = BlockHashMap()

    /// Cache statistics.
    public private(set) var stats = CacheStats()

    // MARK: - Initialization

    /// Create a paged cache manager.
    ///
    /// - Parameters:
    ///   - blockSize: Number of tokens per block (default 64).
    ///   - maxBlocks: Total pool size including the null sentinel (default 1000).
    public init(blockSize: Int = 64, maxBlocks: Int = 1000, modelKey: String? = nil) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.modelKey = modelKey

        // Pre-allocate all blocks.
        self.blocks = (0..<maxBlocks).map { CacheBlock(blockId: $0, blockSize: blockSize) }

        // Block 0 is the null sentinel — never allocate it.
        // Blocks 1..<maxBlocks go into the free queue.
        for i in 1..<maxBlocks {
            freeQueue.append(blocks[i])
        }

        stats.totalBlocks = maxBlocks
        stats.freeBlocks = maxBlocks - 1
    }

    // MARK: - Core Methods

    /// Allocate a block from the free queue.
    ///
    /// - Returns: A freshly-reset ``CacheBlock`` with `refCount == 1`,
    ///   or `nil` if no blocks are available.
    public func allocateBlock() -> CacheBlock? {
        lock.lock()
        defer { lock.unlock() }
        return _allocateBlock()
    }

    /// Internal allocate (caller must hold the lock).
    ///
    /// LRU eviction semantics: the head of the free queue is the least
    /// recently used block. If it still carries a `blockHash` from a previous
    /// store (the "cached but free" state introduced by `storeTokenSequence`
    /// releasing refs to 0 after registration), we evict it from the
    /// hash map here so a future prefix lookup won't return stale data.
    /// Without this, blocks stay pinned at refCount==1 forever and
    /// `allocateBlock` starts returning nil once the pool fills.
    /// Fixes vmlx #68 (paged cache blocks not released on completion).
    private func _allocateBlock() -> CacheBlock? {
        guard let block = freeQueue.popFirst() else { return nil }
        if block.blockHash != nil {
            hashMap.remove(block)
            stats.evictions += 1
        }
        block.reset()
        block.incrementRef()
        stats.allocatedBlocks += 1
        stats.freeBlocks -= 1
        return block
    }

    /// Release a block back to the pool.
    ///
    /// Decrements the reference count; when it reaches zero the block is
    /// removed from the hash map, reset, and returned to the free queue.
    ///
    /// - Parameter block: The block to free.
    public func freeBlock(_ block: CacheBlock) {
        lock.lock()
        defer { lock.unlock() }
        _freeBlock(block)
    }

    /// Internal free (caller must hold the lock).
    private func _freeBlock(_ block: CacheBlock) {
        block.decrementRef()
        if block.refCount == 0 {
            hashMap.remove(block)
            block.reset()
            freeQueue.append(block)
            stats.allocatedBlocks -= 1
            stats.freeBlocks += 1
        }
    }

    /// Look up a cached block by its content hash.
    ///
    /// Updates hit/miss statistics.
    ///
    /// - Parameter hash: The chain hash to look up.
    /// - Returns: The matching ``CacheBlock``, or `nil` on a miss.
    public func findCachedBlock(hash: String) -> CacheBlock? {
        lock.lock()
        defer { lock.unlock() }
        return _findCachedBlock(hash: hash)
    }

    /// Internal find (caller must hold the lock).
    private func _findCachedBlock(hash: String) -> CacheBlock? {
        if let block = hashMap.find(hash: hash) {
            stats.cacheHits += 1
            return block
        }
        stats.cacheMisses += 1
        return nil
    }

    /// Register a block in the hash map under the given hash.
    ///
    /// - Parameters:
    ///   - block: The block to register.
    ///   - hash: The chain hash to use as the key.
    public func registerBlock(_ block: CacheBlock, hash: String) {
        lock.lock()
        defer { lock.unlock() }
        block.blockHash = hash
        hashMap.insert(block)
    }

    /// Reset the manager to its initial state, freeing all blocks.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        hashMap.removeAll()

        // Drain the free queue so we can rebuild it cleanly.
        while freeQueue.popFirst() != nil {}

        for i in 0..<maxBlocks {
            blocks[i].reset()
        }

        // Re-enqueue blocks 1..<maxBlocks.
        for i in 1..<maxBlocks {
            freeQueue.append(blocks[i])
        }

        stats = CacheStats()
        stats.totalBlocks = maxBlocks
        stats.freeBlocks = maxBlocks - 1
    }

    // MARK: - Prefix Matching

    /// Store a token sequence as a chain of cache blocks.
    ///
    /// Tokens are split into ``blockSize``-sized chunks. Each chunk's chain
    /// hash incorporates the previous block's hash so that identical prefixes
    /// always produce the same hash chain. Chunks whose hash already exists in
    /// the cache are skipped (prefix sharing).
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence.
    ///   - layerData: Per-chunk, per-layer KV tensors. `layerData[chunkIndex]`
    ///     contains one `(keys, values)` tuple per transformer layer.
    public func storeTokenSequence(
        tokens: [Int],
        layerData: [[(keys: MLXArray, values: MLXArray)]],
        mediaSalt: String? = nil
    ) {
        lock.lock()
        defer { lock.unlock() }

        var parentHash: String? = nil
        var chunkIndex = 0
        var offset = 0

        while offset + blockSize <= tokens.count {
            let chunk = Array(tokens[offset..<(offset + blockSize)])
            let hash = CacheBlock.computeBlockHash(
                parentHash: parentHash, tokenIds: chunk,
                modelKey: modelKey, mediaSalt: mediaSalt)

            // Skip if this block already exists in the cache.
            if hashMap.find(hash: hash) != nil {
                parentHash = hash
                offset += blockSize
                chunkIndex += 1
                continue
            }

            // Allocate a new block.
            guard let block = _allocateBlock() else { break }

            block.tokenIds = chunk
            if chunkIndex < layerData.count {
                block.cacheData = layerData[chunkIndex].map { Optional($0) }
            }

            block.blockHash = hash
            hashMap.insert(block)

            // "Cached but free" transition: the block is now populated and
            // registered in the hash map for future prefix matches, but it
            // is not actively referenced by any live generation. Drop the
            // store-time ref and re-enqueue at the LRU tail so it is both
            // still findable AND reclaimable under memory pressure. Without
            // this, blocks stay at refCount==1 forever, the free queue
            // drains, and the cache silently stops accepting new blocks.
            // Fixes vmlx #68.
            block.decrementRef()
            stats.allocatedBlocks -= 1
            stats.freeBlocks += 1
            freeQueue.append(block)

            parentHash = hash
            offset += blockSize
            chunkIndex += 1
        }
    }

    /// Fetch the longest cached prefix for a token sequence.
    ///
    /// Walks the tokens in ``blockSize``-sized chunks, computing chain hashes
    /// and looking up each in the hash map. Stops at the first miss.
    ///
    /// - Parameter tokens: The full token sequence to match.
    /// - Returns: A ``PrefixFetchResult`` describing how many tokens matched
    ///   and which blocks hold the cached data, or `nil` if no prefix matches.
    public func fetchPrefix(tokens: [Int], mediaSalt: String? = nil) -> PrefixFetchResult? {
        lock.lock()
        defer { lock.unlock() }

        var parentHash: String? = nil
        var matchedBlocks: [CacheBlock] = []
        var offset = 0

        while offset + blockSize <= tokens.count {
            let chunk = Array(tokens[offset..<(offset + blockSize)])
            let hash = CacheBlock.computeBlockHash(
                parentHash: parentHash, tokenIds: chunk,
                modelKey: modelKey, mediaSalt: mediaSalt)

            if let block = _findCachedBlock(hash: hash) {
                // Mark as most-recently-used so subsequent evictions reclaim
                // cold blocks first. Safe because the block is cached-but-free
                // and still resident in the freeQueue. Part of vmlx #68 fix.
                freeQueue.touch(block)
                matchedBlocks.append(block)
                parentHash = hash
                offset += blockSize
            } else {
                break
            }
        }

        guard !matchedBlocks.isEmpty else { return nil }

        let matchedTokens = matchedBlocks.count * blockSize
        let remainingTokens = Array(tokens[offset...])

        return PrefixFetchResult(
            matchedTokens: matchedTokens,
            remainingTokens: remainingTokens,
            blocks: matchedBlocks
        )
    }
}
