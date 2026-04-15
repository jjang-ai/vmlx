// Copyright © 2024 Apple Inc.

import CryptoKit
import Foundation
import MLX

/// A single block in the paged KV cache system.
///
/// Each `CacheBlock` stores up to `blockSize` tokens worth of KV cache data
/// along with a chain hash for prefix matching. Blocks are reference-counted
/// so that shared prefixes can be reused across multiple sequences.
public final class CacheBlock: @unchecked Sendable {

    // MARK: - Properties

    /// Unique block ID in the pool.
    public let blockId: Int

    /// Maximum number of tokens this block can hold.
    public let blockSize: Int

    /// SHA-256 chain hash for prefix matching. The hash incorporates the
    /// parent block's hash so that identical token sequences rooted at the
    /// same prefix always produce the same hash.
    public internal(set) var blockHash: String?

    /// Token IDs stored in this block (up to ``blockSize``).
    public internal(set) var tokenIds: [Int]

    /// Per-layer KV tensors. Each element corresponds to one transformer
    /// layer; `nil` entries represent SSM/non-attention layers.
    public internal(set) var cacheData: [(keys: MLXArray, values: MLXArray)?]?

    /// Reference count for shared prefix blocks.
    public private(set) var refCount: Int

    // MARK: - Computed Properties

    /// Number of tokens currently stored.
    public var tokenCount: Int { tokenIds.count }

    /// Whether the block is full (has reached ``blockSize``).
    public var isFull: Bool { tokenCount >= blockSize }

    /// Whether the block contains no tokens.
    public var isEmpty: Bool { tokenIds.isEmpty }

    // MARK: - Initialization

    /// Creates a new cache block.
    /// - Parameters:
    ///   - blockId: Unique identifier for this block in the pool.
    ///   - blockSize: Maximum number of tokens the block can hold.
    public init(blockId: Int, blockSize: Int) {
        self.blockId = blockId
        self.blockSize = blockSize
        self.blockHash = nil
        self.tokenIds = []
        self.cacheData = nil
        self.refCount = 0
    }

    // MARK: - Reference Counting

    /// Increment the reference count by one.
    public func incrementRef() {
        refCount += 1
    }

    /// Decrement the reference count by one, flooring at zero.
    public func decrementRef() {
        refCount = max(0, refCount - 1)
    }

    // MARK: - Reset

    /// Reset the block to its initial empty state, clearing all stored data.
    public func reset() {
        blockHash = nil
        tokenIds = []
        cacheData = nil
        refCount = 0
    }

    // MARK: - Hashing

    /// Compute a deterministic chain hash from a parent hash and token IDs.
    ///
    /// The hash is computed as `SHA-256(parentHashBytes || tokenIdsRawBytes)`
    /// where `parentHashBytes` is the UTF-8 encoding of the parent hash string
    /// (empty if `nil`), and `tokenIdsRawBytes` is the raw memory representation
    /// of the token ID array.
    ///
    /// - Parameters:
    ///   - parentHash: The hash of the preceding block in the chain, or `nil` for the first block.
    ///   - tokenIds: The token IDs to include in the hash.
    /// - Returns: A 64-character lowercase hex string.
    public static func computeBlockHash(
        parentHash: String?,
        tokenIds: [Int],
        modelKey: String? = nil,
        mediaSalt: String? = nil
    ) -> String {
        var hasher = SHA256()

        // Feed model key first (prevents cross-model cache poisoning)
        if let modelKey {
            hasher.update(data: Data(modelKey.utf8))
        }

        // Feed media salt before tokens so VLM inputs with the same text
        // prefix but different images/videos get distinct block hashes.
        // Passing `nil` preserves the exact hash of the pre-existing
        // text-only path (byte-for-byte backward compatible).
        if let mediaSalt {
            hasher.update(data: Data("|media:".utf8))
            hasher.update(data: Data(mediaSalt.utf8))
        }

        // Feed parent hash bytes (if any)
        if let parentHash {
            let parentData = Data(parentHash.utf8)
            hasher.update(data: parentData)
        }

        // Feed token IDs as raw bytes
        tokenIds.withUnsafeBufferPointer { buffer in
            let rawBuffer = UnsafeRawBufferPointer(buffer)
            hasher.update(bufferPointer: rawBuffer)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
