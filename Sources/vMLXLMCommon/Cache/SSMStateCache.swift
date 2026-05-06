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

    /// Number of times the hybrid-SSM re-derive watcher fired (iter-40).
    /// Incremented by `markReDeriveFired()` from `maybeReDeriveSSMState`
    /// whenever a thinking-template turn triggers a prompt-only pass.
    /// Distinct from `misses` so the UI can distinguish "I never tried"
    /// from "I tried but the cache was contaminated so I re-derived".
    /// Zero for pure-attention models (re-derive always no-ops early).
    public private(set) var reDerives: Int = 0

    /// Bump the re-derive counter. Lock-protected so it stays
    /// consistent with the hits/misses counters under concurrent
    /// access. Called from the detached Task in `Stream.swift` or
    /// synchronously when `VMLX_DISABLE_SSM_RE_DERIVE_ASYNC=1`.
    public func markReDeriveFired() {
        lock.lock()
        reDerives &+= 1
        lock.unlock()
    }

    // MARK: - Initialization

    /// Creates a new SSM state cache.
    /// - Parameters:
    ///   - maxEntries: Maximum number of entries before LRU eviction
    ///     kicks in. Defaults to 50.
    ///   - modelKey: Stable identifier for the loaded model. Mixed into
    ///     every key so hot-swapping models in-process can't collide on
    ///     the same prompt prefix. Empty/`nil` = pre-fix parity.
    public init(maxEntries: Int = 50, modelKey: String? = nil) {
        self.maxEntries = maxEntries
        self.modelKey = modelKey
        self.entries = []
    }

    /// Mixed into every cache key. Set once at init time by
    /// `CacheCoordinator` based on `CacheCoordinatorConfig.modelKey`.
    private let modelKey: String?

    /// §441 — Optional disk tier for cold-start cache survival. When
    /// non-nil, `store()` write-throughs to disk and `fetchEntry()`
    /// falls through on memory miss + reloads into the LRU slot.
    /// Wired by `CacheCoordinator` when
    /// `enableSSMCompanionDiskCache` setting is on. Default nil =
    /// in-memory only (pre-§441 behavior).
    public var diskStore: SSMCompanionDiskStore? = nil

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
        let key = Self.makeKey(tokens: tokens, boundary: boundary, mediaSalt: mediaSalt, modelKey: modelKey)

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

        // §441 — write-through to disk tier if configured. Done outside
        // the in-memory critical path's hot section by capturing the
        // values we need first; the disk write itself takes the
        // SSMCompanionDiskStore's own lock. Failures are swallowed so a
        // disk-full / permission error never breaks generation — the
        // in-memory entry above is still authoritative.
        //
        // Iter 143: thread `mediaSalt` through to the disk write so VL/
        // Omni hybrid prefixes don't collide with text-only prefixes
        // sharing the same token sequence. Previously hardcoded nil
        // here → silent L2 alias on cold start of the next session.
        if let disk = diskStore {
            try? disk.store(
                ssmStates: copies,
                tokens: tokens,
                boundary: boundary,
                mediaSalt: mediaSalt,
                isComplete: isComplete)
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
        let key = Self.makeKey(tokens: tokens, boundary: boundary, mediaSalt: mediaSalt, modelKey: modelKey)

        lock.lock()
        defer { lock.unlock() }

        guard let index = entries.firstIndex(where: { $0.key == key }) else {
            misses += 1
            // §441 — fall through to disk tier on memory miss. Reload
            // into the LRU slot so subsequent fetches in this session
            // hit memory. Disk fetch returns deep-copied arrays from
            // safetensors deserialize, so they're already detached
            // from any lazy graph.
            if let disk = diskStore,
               let result = disk.fetch(
                   tokens: tokens, boundary: boundary, mediaSalt: mediaSalt)
            {
                // Hydrate into LRU. Subtract the miss we just bumped
                // because the disk lookup found it — this is a hit
                // from the user's perspective, just one tier deeper.
                misses -= 1
                hits += 1
                entries.append((
                    key: key,
                    states: result.states,
                    isComplete: result.isComplete))
                if entries.count > maxEntries { entries.removeFirst() }
                // Return another deep copy so the caller can mutate
                // freely — same contract as the in-memory hit path.
                let safeCopies = result.states.map { $0 * 1 }
                return FetchResult(states: safeCopies, isComplete: result.isComplete)
            }
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

    /// Check whether an in-memory companion snapshot exists without
    /// incrementing hit/miss counters or deep-copying state arrays.
    public func contains(
        tokens: [Int], boundary: Int, mediaSalt: String? = nil
    ) -> Bool {
        let key = Self.makeKey(tokens: tokens, boundary: boundary, mediaSalt: mediaSalt, modelKey: modelKey)

        lock.lock()
        defer { lock.unlock() }

        return entries.contains { $0.key == key && !$0.states.isEmpty }
    }

    /// Remove all entries and reset hit/miss/re-derive statistics.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        entries.removeAll()
        hits = 0
        misses = 0
        reDerives = 0
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
        tokens: [Int], boundary: Int, mediaSalt: String? = nil,
        modelKey: String? = nil
    ) -> String {
        // P0-2 (2026-04-30): Converged hash formula. Previously this hashed
        // raw Int bytes via `UnsafeRawBufferPointer(prefix)` which differed
        // from `SSMCompanionDiskStore.keyFor` (raw bytes too, but with `:`
        // separator) AND from Python's
        //   `model_key + "\x00" + json.dumps(tokens[:N], separators=(",",":"))`
        // (audit doc AUDIT-SSM-WARMPASS-PARITY.md §1).
        // Result: L1 vs L2 mismatch → disk backfill never lands in L1.
        // Python ↔ Swift mismatch → no future cross-process sharing.
        // The new shared formula:
        //   SHA-256( modelKey-blob || mediaSalt-blob || "|tokens:" || json.dumps([…]) )
        // matches Python byte-for-byte once modelKey and mediaSalt are blank
        // (the "|tokens:" tag + JSON-string-of-list is identical to Python
        // when the auxiliary salts are empty).
        let prefix = Array(tokens.prefix(boundary))
        var hasher = SHA256()

        // Mix modelKey FIRST so hot-swapping models in-process can't
        // collide. Mirrors `DiskCache.hashTokens` / `CacheBlock.computeBlockHash`.
        if let mk = modelKey, !mk.isEmpty {
            hasher.update(data: Data("|model:".utf8))
            hasher.update(data: Data(mk.utf8))
        }
        // VLM mediaSalt — also unconditionally mixed so L2 disk store agrees.
        if let salt = mediaSalt, !salt.isEmpty {
            hasher.update(data: Data("|media:".utf8))
            hasher.update(data: Data(salt.utf8))
        }
        // Tokens encoded as JSON array string with no whitespace,
        // matching Python `json.dumps([…], separators=(",",":"))` byte-for-byte.
        hasher.update(data: Data("|tokens:".utf8))
        hasher.update(data: Self.jsonEncodeIntList(prefix))

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Encode an Int array as `[a,b,c]` UTF-8 bytes — byte-identical to
    /// Python `json.dumps(list, separators=(",",":"))` for any list of
    /// non-negative integers (token IDs).
    static func jsonEncodeIntList(_ ints: [Int]) -> Data {
        var s = "["
        for (i, x) in ints.enumerated() {
            if i > 0 { s.append(",") }
            s.append(String(x))
        }
        s.append("]")
        return Data(s.utf8)
    }
}
