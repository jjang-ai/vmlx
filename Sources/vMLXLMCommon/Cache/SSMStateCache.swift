// Copyright © 2024 Apple Inc.

import CryptoKit
import Foundation
import MLX
import os

/// An LRU companion cache for SSM layer state in hybrid models
/// (Nemotron-H, Qwen3.5-A3B, Jamba).
///
/// SSM state is cumulative (path-dependent) and cannot be reconstructed
/// from KV cache alone, so it must be cached separately. Entries are keyed
/// by a SHA-256 hash of the token prefix up to a given boundary, and the
/// cache uses LRU eviction when the entry limit is reached.
///
/// All public methods are thread-safe via `OSAllocatedUnfairLock`.
///
/// **Deep-copy semantics**: ``fetch(tokens:boundary:)`` returns independent
/// copies of the stored state arrays because model forward passes modify
/// SSM state in-place; sharing would corrupt the cached snapshot.
public final class SSMStateCache: @unchecked Sendable {

    // MARK: - Properties

    private let lock = OSAllocatedUnfairLock()
    private let maxEntries: Int
    private var entries: [(key: String, states: [MLXArray], isComplete: Bool)]

    /// One fetched SSM-cache entry. `isComplete == true` means the entry
    /// captured the full prompt prefix at boundary and is safe to extend
    /// (the next turn can apply additional tokens past the boundary).
    /// `isComplete == false` is a partial capture (mid-stream snapshot
    /// after only some of the prompt tokens were processed) and callers
    /// MUST NOT extend it — they should re-derive instead.
    ///
    /// Mirrors Python's `(states, is_complete)` tuple from
    /// `vmlx_engine/utils/ssm_companion_cache.py`.
    public struct FetchResult: @unchecked Sendable {
        public let states: [MLXArray]
        public let isComplete: Bool
        public init(states: [MLXArray], isComplete: Bool) {
            self.states = states
            self.isComplete = isComplete
        }
    }

    /// Number of successful cache hits since creation (or last ``clear()``).
    public private(set) var hits: Int = 0

    /// Number of cache misses since creation (or last ``clear()``).
    public private(set) var misses: Int = 0

    // MARK: - Initialization

    /// Creates a new SSM state cache.
    /// - Parameter maxEntries: Maximum number of entries before LRU eviction
    ///   kicks in. Defaults to 50.
    public init(maxEntries: Int = 50) {
        self.maxEntries = maxEntries
        self.entries = []
    }

    // MARK: - Public API

    /// Store SSM layer states for a given token prefix.
    ///
    /// Each state array is materialized (evaluated) immediately so that the
    /// stored snapshot is independent of the lazy computation graph.
    ///
    /// - Parameters:
    ///   - ssmStates: The per-layer SSM state arrays to cache.
    ///   - tokens: The full token sequence for the current generation.
    ///   - boundary: The number of tokens (from the start) to include in the
    ///     cache key.
    public func store(
        ssmStates: [MLXArray],
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        isComplete: Bool = true
    ) {
        let key = Self.makeKey(tokens: tokens, boundary: boundary, mediaSalt: mediaSalt)

        lock.lock()
        defer { lock.unlock() }

        // Remove existing entry with same key (if any)
        entries.removeAll { $0.key == key }

        // Materialize each state array into a FRESH buffer. The prior
        // `arr[.ellipsis]` was a shape-preserving slice view that shared
        // the source tensor's storage, so any in-place mutation of the
        // source (e.g. a subsequent Mamba step that writes back into the
        // same backing array) would silently corrupt the cached entry.
        // `arr * 1` forces materialization into a new buffer on the
        // current stream; subsequent MLX materialization detaches from
        // the lazy graph. Historical precedent: "fetch returned shared
        // refs → model mutated in-place → deep-copy fix" from session
        // 2026-03-28b.
        let copies = ssmStates.map { arr -> MLXArray in
            let copy = arr * 1
            MLX.eval(copy)
            return copy
        }

        // Append to end (most recently used position)
        entries.append((key: key, states: copies, isComplete: isComplete))

        // Evict oldest if over capacity
        if entries.count > maxEntries {
            entries.removeFirst()
        }
    }

    /// Fetch cached SSM states for a given token prefix.
    ///
    /// Returns deep copies of the stored arrays so that in-place mutations
    /// during model forward passes do not corrupt the cache.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence for the current generation.
    ///   - boundary: The number of tokens (from the start) to include in the
    ///     cache key.
    /// - Returns: Deep copies of the cached state arrays, or `nil` on a miss.
    /// Backward-compatible shim: forwards to `fetchEntry` and discards
    /// the completeness flag. New callers should prefer `fetchEntry`.
    public func fetch(
        tokens: [Int], boundary: Int, mediaSalt: String? = nil
    ) -> [MLXArray]? {
        fetchEntry(tokens: tokens, boundary: boundary, mediaSalt: mediaSalt)?.states
    }

    /// Fetch with completeness flag. Returns `FetchResult.isComplete=false`
    /// for partial-prefix entries that callers must NOT extend (re-derive
    /// instead). Mirrors Python's `(states, is_complete)` tuple semantics
    /// from `vmlx_engine/utils/ssm_companion_cache.py`.
    public func fetchEntry(
        tokens: [Int], boundary: Int, mediaSalt: String? = nil
    ) -> FetchResult? {
        let key = Self.makeKey(tokens: tokens, boundary: boundary, mediaSalt: mediaSalt)

        lock.lock()
        defer { lock.unlock() }

        guard let index = entries.firstIndex(where: { $0.key == key }) else {
            misses += 1
            return nil
        }

        let entry = entries[index]

        // Empty states array is treated as a miss (bug fix from osa-jang ba07392)
        guard !entry.states.isEmpty else {
            misses += 1
            return nil
        }

        // LRU touch: move to end
        entries.remove(at: index)
        entries.append(entry)

        hits += 1

        // Return deep copies — model forward passes modify SSM state
        // in-place, so the fetched arrays must NOT share storage with
        // the cached entry. Same reasoning as the store path above:
        // `[.ellipsis]` would be a view, `* 1` forces a fresh buffer.
        let copies = entry.states.map { $0 * 1 }
        return FetchResult(states: copies, isComplete: entry.isComplete)
    }

    /// Remove all entries and reset hit/miss statistics.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        entries.removeAll()
        hits = 0
        misses = 0
    }

    // MARK: - Key Generation

    /// Compute a deterministic cache key from the first `boundary` tokens.
    ///
    /// The key is the SHA-256 hash of the raw bytes of `tokens[0..<boundary]`,
    /// returned as a 64-character lowercase hex string.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence.
    ///   - boundary: How many tokens from the start to include in the hash.
    /// - Returns: A 64-character lowercase hex string.
    public static func makeKey(
        tokens: [Int], boundary: Int, mediaSalt: String? = nil
    ) -> String {
        let prefix = Array(tokens.prefix(boundary))
        var hasher = SHA256()

        // Audit F-G1 (P0): mix mediaSalt BEFORE tokens so VLM multi-turn
        // requests with different images but the same text prefix produce
        // different keys. The paged L1 and disk L2 tiers already do this
        // via `CacheBlock.computeBlockHash` and `DiskCache.hashTokens` —
        // the SSM companion tier was the only cache that hashed tokens
        // alone, so it would HIT on image-B requests with stored
        // image-A state and silently corrupt generation. Empty salt is
        // the text-only path and hashes the same as pre-fix entries.
        if let salt = mediaSalt, !salt.isEmpty {
            hasher.update(data: Data("|media:".utf8))
            hasher.update(data: Data(salt.utf8))
        }

        prefix.withUnsafeBufferPointer { buffer in
            let rawBuffer = UnsafeRawBufferPointer(buffer)
            hasher.update(bufferPointer: rawBuffer)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
