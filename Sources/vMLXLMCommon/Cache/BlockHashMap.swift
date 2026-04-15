// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation

/// O(1) dictionary wrapper mapping content hashes to ``CacheBlock`` instances.
///
/// Used by the paged KV cache to detect prefix sharing: when a new sequence
/// produces the same token block as an existing one, the hash map lets us
/// find and reuse the already-computed block instead of recomputing it.
public final class BlockHashMap: @unchecked Sendable {

    // MARK: - Storage

    private var map: [String: CacheBlock] = [:]

    // MARK: - Init

    public init() {}

    // MARK: - API

    /// Insert a block keyed by its ``CacheBlock/blockHash``.
    /// No-op if the block's hash is `nil`.
    public func insert(_ block: CacheBlock) {
        guard let hash = block.blockHash else { return }
        map[hash] = block
    }

    /// Look up a block by its content hash.
    public func find(hash: String) -> CacheBlock? {
        map[hash]
    }

    /// Remove the entry for a block (keyed by its ``CacheBlock/blockHash``).
    /// No-op if the block's hash is `nil` or not present.
    public func remove(_ block: CacheBlock) {
        guard let hash = block.blockHash else { return }
        map.removeValue(forKey: hash)
    }

    /// Remove all entries.
    public func removeAll() {
        map.removeAll()
    }

    /// The number of entries currently stored.
    public var count: Int {
        map.count
    }
}
