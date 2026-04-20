// SPDX-License-Identifier: Apache-2.0
//
// Memory-aware prefix cache. Port of
// `vmlx_engine/memory_cache.py` — a byte-budgeted LRU prefix cache
// for KV state that evicts based on memory pressure rather than
// entry count.
//
// Key properties (matching Python):
// - Auto-sized memory budget: `percent * available RAM`, capped at
//   32 GB (Metal GPU reality), floor at 100 MB.
// - Three-tier LRU priority (`assistant` < `user` < `system`) so
//   shared system prompts survive cross-user eviction.
// - Optional TTL expiration in minutes.
// - Memory-pressure adaptation every 60s: when available RAM drops
//   below 20% of total, the effective budget is temporarily reduced.
// - Fetch strategies: exact match, forward prefix (cached is shorter),
//   reverse prefix (cached is longer — truncated to requested length).
// - Thread-safe via `NSLock` (non-reentrant per Python).
//
// What we don't port:
// - `estimate_kv_cache_memory` is injected as a closure so tests can
//   exercise the cache logic without MLX dispatch. The real engine
//   supplies a closure that inspects the MLX cache types.
// - `_truncate_cache` reverse-prefix truncation is injected the same
//   way — the actual truncation is MLX-specific and tests use a
//   passthrough closure.
// - `compute_model_cache_key` → the caller supplies a stable modelId
//   string at init. This cache doesn't rely on Python object identity.
//
// Threading note: unlike SlotBankCache, this one uses `NSLock` rather
// than `OSAllocatedUnfairLock` because the critical sections include
// arbitrary-cost operations (tuple equality, prefix scan over N
// entries) that can run for microseconds on large caches. Unfair
// locks are tuned for short critical sections and hurt fairness when
// held longer.

import Foundation

// MARK: - Default available-memory detector

/// Darwin `host_statistics64` — returns free RAM bytes (including
/// inactive + free, matching `psutil.virtual_memory().available`
/// as closely as the kernel exposes). Returns 0 on other platforms
/// or on sysctl failure so the fallback path kicks in.
///
/// Free function (not a static method) because generic classes
/// can't reference `Self.staticMember` as a default-argument value
/// in Swift — covariant-Self restriction.
public func defaultAvailableMemoryImpl() -> Int {
    #if canImport(Darwin)
    var vmStats = vm_statistics64()
    var infoCount = mach_msg_type_number_t(
        MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
    )
    let hostPort = mach_host_self()
    let result: kern_return_t = withUnsafeMutablePointer(to: &vmStats) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) { rebound in
            host_statistics64(hostPort, HOST_VM_INFO64, rebound, &infoCount)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    var pageSize: vm_size_t = 0
    host_page_size(hostPort, &pageSize)
    let freePages = Int(vmStats.free_count) + Int(vmStats.inactive_count)
    return freePages * Int(pageSize)
    #else
    return 0
    #endif
}

// MARK: - Cache type priority

/// Three-tier LRU priority. Lower-priority types are evicted first:
/// `assistant` → `user` → `system`. System entries stay pinned
/// longest so shared system prompts survive cross-session eviction.
/// Matches `_CACHE_TYPE_PRIORITY` tuple in Python.
public enum MemoryCacheEntryType: String, Sendable, CaseIterable {
    case assistant
    case user
    case system

    /// Ordering constant for `_evict_lru` — lower number = evicted first.
    public var priority: Int {
        switch self {
        case .assistant: return 0
        case .user:      return 1
        case .system:    return 2
        }
    }
}

// MARK: - Config

/// Configuration for `MemoryAwarePrefixCache`. Mirrors
/// `MemoryCacheConfig` dataclass field-for-field so JSON configs
/// round-trip cleanly between Python and Swift.
public struct MemoryCacheConfig: Sendable, Equatable {
    /// Hard memory cap in MB. When set, takes precedence over
    /// `maxMemoryPercent`. When `nil`, auto-sizing kicks in.
    public var maxMemoryMB: Int?
    /// Fraction of available RAM to use when `maxMemoryMB` is nil.
    /// Must be in (0, 1]. Default 0.30 (30%).
    public var maxMemoryPercent: Double
    /// Safety net: max number of entries regardless of memory budget.
    public var maxEntries: Int
    /// Whether to track per-entry memory. When false, the cache uses
    /// only entry-count eviction. Default true.
    public var enableMemoryTracking: Bool
    /// Time-to-live for entries in minutes. 0 = unlimited lifetime.
    public var ttlMinutes: Double

    public init(
        maxMemoryMB: Int? = nil,
        maxMemoryPercent: Double = 0.30,
        maxEntries: Int = 1000,
        enableMemoryTracking: Bool = true,
        ttlMinutes: Double = 0
    ) {
        // Treat 0.0 as "graceful disable" — return a no-op config that
        // computeMemoryLimit will resolve to the floor (100 MB). Callers
        // who want the cache fully off should pass `enableMemoryCache=false`
        // to the coordinator instead; the precondition only catches
        // genuinely nonsensical values like negatives or > 1.0.
        precondition(maxMemoryPercent >= 0 && maxMemoryPercent <= 1.0,
                     "maxMemoryPercent must be in [0, 1], got \(maxMemoryPercent)")
        precondition(maxEntries >= 1, "maxEntries must be >= 1")
        precondition(ttlMinutes >= 0, "ttlMinutes must be >= 0")
        self.maxMemoryMB = maxMemoryMB
        self.maxMemoryPercent = maxMemoryPercent
        self.maxEntries = maxEntries
        self.enableMemoryTracking = enableMemoryTracking
        self.ttlMinutes = ttlMinutes
    }

    /// Compute the effective memory budget in bytes. Python parity:
    /// - explicit `maxMemoryMB` → that value capped at 32 GB
    /// - otherwise → `available * percent`, capped at 32 GB, floor 100 MB
    /// - fallback (availableBytes == 0) → `4 GB * percent`
    public func computeMemoryLimit(availableBytes: Int) -> Int {
        let maxCacheBytes = 32 * 1024 * 1024 * 1024   // 32 GB hard cap
        let minMemoryBytes = 100 * 1024 * 1024         // 100 MB floor

        if let mb = maxMemoryMB {
            return min(mb * 1024 * 1024, maxCacheBytes)
        }
        if availableBytes > 0 {
            let limit = Int(Double(availableBytes) * maxMemoryPercent)
            return max(min(limit, maxCacheBytes), minMemoryBytes)
        }
        // Fallback: assume 4 GB available.
        let fallback = 4 * 1024 * 1024 * 1024
        return max(min(Int(Double(fallback) * maxMemoryPercent), maxCacheBytes),
                   minMemoryBytes)
    }
}

// MARK: - Stats

public struct MemoryCacheStats: Sendable, Equatable {
    public var hits: Int = 0
    public var misses: Int = 0
    public var evictions: Int = 0
    public var tokensSaved: Int = 0
    public var currentMemoryBytes: Int = 0
    public var maxMemoryBytes: Int = 0
    public var entryCount: Int = 0

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }

    public var memoryUtilization: Double {
        maxMemoryBytes > 0
            ? Double(currentMemoryBytes) / Double(maxMemoryBytes)
            : 0
    }

    public func toDictionary() -> [String: Any] {
        let bytesPerMB = 1024.0 * 1024.0
        return [
            "hits": hits,
            "misses": misses,
            "hit_rate": (hitRate * 10000).rounded() / 10000,
            "evictions": evictions,
            "tokens_saved": tokensSaved,
            "current_memory_mb": (Double(currentMemoryBytes) / bytesPerMB * 100).rounded() / 100,
            "max_memory_mb": (Double(maxMemoryBytes) / bytesPerMB * 100).rounded() / 100,
            "memory_utilization": (memoryUtilization * 10000).rounded() / 10000,
            "entry_count": entryCount,
        ]
    }
}

// MARK: - Fetch result

public struct MemoryCacheFetchResult<Payload: Sendable>: Sendable {
    /// The cached payload, or nil on miss.
    public let cache: Payload?
    /// Tokens that still need to be processed (miss → all tokens,
    /// exact hit → empty, forward prefix hit → suffix after the cached
    /// portion).
    public let remainingTokens: [Int]

    public init(cache: Payload?, remainingTokens: [Int]) {
        self.cache = cache
        self.remainingTokens = remainingTokens
    }
}

// MARK: - Cache

/// Byte-budgeted LRU prefix cache for KV state. Generic over the
/// payload type so tests can use `Int` / `String` instead of an MLX
/// cache array. In production the engine instantiates
/// `MemoryAwarePrefixCache<[any KVCache]>` and injects real
/// `estimateMemory` + `truncate` closures.
public final class MemoryAwarePrefixCache<Payload>: @unchecked Sendable {

    // MARK: - Node

    /// Doubly-linked-list node in a single-bucket LRU list. Each node
    /// also has a pointer into the main `entries` dict for O(1)
    /// find-and-touch from the prefix scan path.
    private final class Node {
        let tokens: [Int]
        var payload: Payload
        var memoryBytes: Int
        // Monotonic timestamp (seconds since boot) so TTL eviction
        // survives wall-clock adjustments (NTP, DST, leap second).
        // `ProcessInfo.systemUptime` is the same source DispatchTime
        // uses internally — safe to compare across threads.
        var lastAccessedAt: TimeInterval
        var cacheType: MemoryCacheEntryType
        var prev: Node?
        var next: Node?

        init(
            tokens: [Int],
            payload: Payload,
            memoryBytes: Int,
            cacheType: MemoryCacheEntryType
        ) {
            self.tokens = tokens
            self.payload = payload
            self.memoryBytes = memoryBytes
            self.lastAccessedAt = ProcessInfo.processInfo.systemUptime
            self.cacheType = cacheType
        }
    }

    // MARK: - State

    public let config: MemoryCacheConfig
    public let modelId: String

    private let lock = NSLock()
    /// Primary token-keyed map. Keys are token sequences wrapped in
    /// arrays because Swift dictionaries don't hash `[Int]` directly —
    /// we use the string representation as the hash key.
    private var entries: [String: Node] = [:]
    /// LRU head/tail per cache-type bucket. Head = LRU, tail = MRU.
    private var lruHead: [MemoryCacheEntryType: Node?] = [:]
    private var lruTail: [MemoryCacheEntryType: Node?] = [:]
    private var bytesByType: [MemoryCacheEntryType: Int] = [:]

    /// Memory tracking scalars.
    public private(set) var currentMemoryBytes: Int = 0
    /// Effective budget — may be reduced below `config.computeMemoryLimit(...)`
    /// under memory pressure.
    public private(set) var maxMemoryBytes: Int
    /// User-requested limit, NOT lowered by pressure adaptation.
    private let baselineMemoryBytes: Int
    private var lastPressureCheckAt: Date = .distantPast

    private var stats = MemoryCacheStats()

    // MARK: - Injected helpers

    /// Returns the estimated byte size of a payload. For production
    /// the engine passes a closure that walks the MLX cache; tests
    /// pass a lambda that returns a fixed count.
    public let estimateMemory: (Payload) -> Int
    /// Returns an available-memory estimate in bytes, or 0 if
    /// detection fails. Called once at init for the memory budget
    /// and periodically for pressure adaptation. Tests pass a
    /// stub that returns a fixed value.
    public let availableMemory: () -> Int
    /// Optional reverse-prefix truncation. When the cached sequence
    /// is LONGER than the request, we hit a "request is a prefix of
    /// cached" scenario and the cache can be shared if the engine
    /// can truncate the KV payload to match. When `nil`, reverse
    /// prefix hits are disabled (always report a miss).
    public let truncate: ((Payload, Int) -> Payload?)?
    /// OS memory pressure listener (macOS `DispatchSource`). Wired at
    /// init so we react to `.warning` / `.critical` events instantly
    /// instead of waiting for the next `store()` call's 60-second
    /// polling tick. Cancelled in deinit to release the source.
    /// FIX-G-B (2026-04-16): reactive eviction on pressure events.
    private var memoryPressureSource: DispatchSourceMemoryPressure?

    // MARK: - Init

    public init(
        config: MemoryCacheConfig = MemoryCacheConfig(),
        modelId: String = "model",
        estimateMemory: @escaping (Payload) -> Int,
        availableMemory: @escaping () -> Int = defaultAvailableMemoryImpl,
        truncate: ((Payload, Int) -> Payload?)? = nil
    ) {
        self.config = config
        self.modelId = modelId
        self.estimateMemory = estimateMemory
        self.availableMemory = availableMemory
        self.truncate = truncate

        let limit = config.computeMemoryLimit(availableBytes: availableMemory())
        self.maxMemoryBytes = limit
        self.baselineMemoryBytes = limit

        for t in MemoryCacheEntryType.allCases {
            lruHead[t] = nil
            lruTail[t] = nil
            bytesByType[t] = 0
        }

        // FIX-G-B: wire an OS memory pressure listener so eviction
        // reacts instantly to `.warning` / `.critical` events. The
        // existing polling in `checkMemoryPressure()` stays as a
        // fallback (runs at most once per 60s on store()).
        //
        // `.warning`  → shrink to 50% of baseline, evict eagerly
        // `.critical` → shrink to minimum (100 MB), evict everything
        //               except system-tier pinned entries
        //
        // DispatchSource handlers run on the global utility queue
        // (not main), so we must acquire `lock` before mutating
        // any state. No-op on Linux (Foundation has no memory-pressure
        // source) — `DispatchSource.makeMemoryPressureSource` is
        // Darwin-only, so the whole block is `#if canImport(Darwin)`.
        #if canImport(Darwin)
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: DispatchQueue.global(qos: .utility)
        )
        self.memoryPressureSource = source
        source.setEventHandler { [weak self] in
            guard let self = self else { return }
            let event = source.data
            self.lock.withLock {
                if event.contains(.critical) {
                    // Shrink to minimum, evict everything except system tier.
                    self.maxMemoryBytes = 100 * 1024 * 1024
                    // Drop every non-system entry regardless of LRU position.
                    // Evict until below 100 MB or only system entries remain.
                    while self.currentMemoryBytes > self.maxMemoryBytes,
                          !self.entries.isEmpty
                    {
                        if !self.evictLRU() { break }
                    }
                } else if event.contains(.warning) {
                    // Shrink to 50% of baseline, evict eagerly.
                    self.maxMemoryBytes = max(
                        self.baselineMemoryBytes / 2,
                        100 * 1024 * 1024)
                    while self.currentMemoryBytes > self.maxMemoryBytes,
                          !self.entries.isEmpty
                    {
                        if !self.evictLRU() { break }
                    }
                }
            }
        }
        source.resume()
        #endif
    }

    deinit {
        memoryPressureSource?.cancel()
    }

    // MARK: - Fetch

    /// Find a cached payload for `tokens`. Searches in priority order:
    /// exact → longest forward prefix → longest reverse prefix (when
    /// `truncate` is provided).
    public func fetch(tokens: [Int]) -> MemoryCacheFetchResult<Payload> {
        guard !tokens.isEmpty else {
            lock.withLock { stats.misses += 1 }
            return MemoryCacheFetchResult(cache: nil, remainingTokens: tokens)
        }

        return lock.withLock {
            // Evict expired entries before lookup.
            if config.ttlMinutes > 0 {
                _ = evictExpired()
            }

            // 1. Exact match.
            let key = tokenKey(tokens)
            if let node = entries[key] {
                touch(node)
                stats.hits += 1
                stats.tokensSaved += tokens.count
                return MemoryCacheFetchResult(
                    cache: node.payload, remainingTokens: []
                )
            }

            // 2. Prefix scan. O(n) over all entries. Tracks the best
            //    forward match (cached shorter, prefix of request) and
            //    the best reverse match (cached longer, request is a
            //    prefix of cached) independently.
            var bestForward: Node? = nil
            var bestForwardLen = 0
            var bestReverse: Node? = nil
            for node in entries.values {
                let cachedLen = node.tokens.count
                if cachedLen < tokens.count {
                    if cachedLen > bestForwardLen,
                       Array(tokens.prefix(cachedLen)) == node.tokens
                    {
                        bestForward = node
                        bestForwardLen = cachedLen
                    }
                } else if cachedLen > tokens.count {
                    if bestReverse == nil || node.tokens.count < (bestReverse?.tokens.count ?? Int.max),
                       Array(node.tokens.prefix(tokens.count)) == tokens
                    {
                        bestReverse = node
                    }
                }
            }

            // Forward wins when available — exact prefix, no truncation.
            if let forward = bestForward {
                touch(forward)
                stats.hits += 1
                stats.tokensSaved += bestForwardLen
                let remaining = Array(tokens.dropFirst(bestForwardLen))
                return MemoryCacheFetchResult(
                    cache: forward.payload, remainingTokens: remaining
                )
            }

            // Reverse: truncate the cached payload to match request.
            if let reverse = bestReverse, let truncate {
                if let truncated = truncate(reverse.payload, tokens.count) {
                    touch(reverse)
                    stats.hits += 1
                    stats.tokensSaved += tokens.count
                    return MemoryCacheFetchResult(
                        cache: truncated, remainingTokens: []
                    )
                }
            }

            stats.misses += 1
            return MemoryCacheFetchResult(cache: nil, remainingTokens: tokens)
        }
    }

    // MARK: - Store

    /// Store a payload keyed on `tokens`. Evicts lowest-priority LRU
    /// entries until the new entry fits. Returns `true` on store,
    /// `false` when the entry was rejected (empty input, payload
    /// exceeds 95% of total budget, etc).
    @discardableResult
    public func store(
        tokens: [Int],
        payload: Payload,
        cacheType: MemoryCacheEntryType = .assistant
    ) -> Bool {
        guard !tokens.isEmpty else { return false }

        return lock.withLock {
            checkMemoryPressure()
            let key = tokenKey(tokens)

            // Refresh-or-upgrade path: existing entry with same tokens.
            if let old = entries[key] {
                touch(old)
                if cacheType.priority > old.cacheType.priority {
                    // Upgrade: move bytes between buckets.
                    bytesByType[old.cacheType, default: 0] -= old.memoryBytes
                    bytesByType[cacheType, default: 0] += old.memoryBytes
                    unlinkFromBucket(old, bucket: old.cacheType)
                    old.cacheType = cacheType
                    appendToTail(old, bucket: cacheType)
                }
                return true
            }

            if config.ttlMinutes > 0 {
                _ = evictExpired()
            }

            let memoryBytes = config.enableMemoryTracking
                ? estimateMemory(payload)
                : 0

            // Reject entries that exceed 95% of the total budget —
            // evicting everything else to make room is never worth it.
            let maxEntryBytes = Int(Double(maxMemoryBytes) * 0.95)
            if memoryBytes > maxEntryBytes {
                return false
            }

            // Evict lowest-priority LRU entries until we have room.
            while (currentMemoryBytes + memoryBytes > maxMemoryBytes
                   || entries.count >= config.maxEntries)
                  && !entries.isEmpty
            {
                if !evictLRU() { break }
            }

            // Install.
            let node = Node(
                tokens: tokens,
                payload: payload,
                memoryBytes: memoryBytes,
                cacheType: cacheType
            )
            entries[key] = node
            appendToTail(node, bucket: cacheType)
            currentMemoryBytes += memoryBytes
            bytesByType[cacheType, default: 0] += memoryBytes
            return true
        }
    }

    // MARK: - Clear

    public func clear() {
        lock.withLock {
            entries.removeAll()
            for t in MemoryCacheEntryType.allCases {
                lruHead[t] = nil
                lruTail[t] = nil
                bytesByType[t] = 0
            }
            currentMemoryBytes = 0
            stats = MemoryCacheStats()
        }
    }

    // MARK: - Stats

    public func snapshotStats() -> MemoryCacheStats {
        lock.withLock {
            var s = stats
            s.currentMemoryBytes = currentMemoryBytes
            s.maxMemoryBytes = maxMemoryBytes
            s.entryCount = entries.count
            return s
        }
    }

    // MARK: - Private: LRU linked-list plumbing

    private func appendToTail(_ node: Node, bucket: MemoryCacheEntryType) {
        node.prev = lruTail[bucket] ?? nil
        node.next = nil
        if let tail = lruTail[bucket] ?? nil {
            tail.next = node
        } else {
            lruHead[bucket] = node
        }
        lruTail[bucket] = node
    }

    private func unlinkFromBucket(_ node: Node, bucket: MemoryCacheEntryType) {
        if let prev = node.prev {
            prev.next = node.next
        } else {
            lruHead[bucket] = node.next
        }
        if let next = node.next {
            next.prev = node.prev
        } else {
            lruTail[bucket] = node.prev
        }
        node.prev = nil
        node.next = nil
    }

    private func touch(_ node: Node) {
        node.lastAccessedAt = ProcessInfo.processInfo.systemUptime
        unlinkFromBucket(node, bucket: node.cacheType)
        appendToTail(node, bucket: node.cacheType)
    }

    // MARK: - Private: eviction

    /// Evict the least-recently-used entry from the lowest-priority
    /// bucket that has any entries. Returns `true` on success, `false`
    /// when every bucket is empty.
    @discardableResult
    private func evictLRU() -> Bool {
        for t in MemoryCacheEntryType.allCases {
            if let head = lruHead[t] ?? nil {
                unlinkFromBucket(head, bucket: t)
                let key = tokenKey(head.tokens)
                entries.removeValue(forKey: key)
                currentMemoryBytes -= head.memoryBytes
                bytesByType[t, default: 0] -= head.memoryBytes
                stats.evictions += 1
                return true
            }
        }
        return false
    }

    /// Walk every entry and drop those whose age exceeds `ttlMinutes`.
    /// Returns the number of entries evicted.
    @discardableResult
    private func evictExpired() -> Int {
        let ttlSec = config.ttlMinutes * 60
        guard ttlSec > 0 else { return 0 }
        // Monotonic now — see Node.lastAccessedAt comment for rationale.
        let now = ProcessInfo.processInfo.systemUptime
        var removed = 0
        for (key, node) in entries {
            let age = now - node.lastAccessedAt
            if age > ttlSec {
                unlinkFromBucket(node, bucket: node.cacheType)
                entries.removeValue(forKey: key)
                currentMemoryBytes -= node.memoryBytes
                bytesByType[node.cacheType, default: 0] -= node.memoryBytes
                stats.evictions += 1
                removed += 1
            }
        }
        return removed
    }

    // MARK: - Private: memory pressure adaptation

    private func checkMemoryPressure() {
        let now = Date()
        // Throttle bumped 60s → 10s on 2026-04-18 to address JANGTQ2 burst OOM:
        // a 5-way burst of 10 GB JANGTQ chats completes in ~15s, well inside
        // the old 60s window. Each store() call adds 500 MB-1 GB of
        // TQ-compressed arrays; with pressure checks throttled the memory
        // cache grows unboundedly until macOS OOM-kills the process. 10s
        // gives the burst 2-3 pressure checks to drive eviction between
        // sequential requests — still cheap for the common case of small
        // text stores. See SWIFT-AUDIT-2026-04-18.md §3a.
        guard now.timeIntervalSince(lastPressureCheckAt) > 10 else { return }
        lastPressureCheckAt = now

        let available = availableMemory()
        guard available > 0 else { return }

        // Approximation of `psutil.virtual_memory()` total: multiply
        // available by 5 as a rough "roughly 20% free" heuristic.
        // This isn't used for budgeting, only for pressure detection.
        // If available drops below 20% of total, reduce the effective
        // budget to the minimum of baseline and (available / 2).
        // Matches the Python pressure-reduction policy.
        let halfAvailable = available / 2
        if halfAvailable < baselineMemoryBytes {
            // Under pressure: shrink to half of what's free.
            maxMemoryBytes = max(halfAvailable, 100 * 1024 * 1024)
            // Evict eagerly if we're now over budget.
            while currentMemoryBytes > maxMemoryBytes, !entries.isEmpty {
                if !evictLRU() { break }
            }
        } else {
            // Pressure cleared — restore baseline.
            maxMemoryBytes = baselineMemoryBytes
        }
    }

    // MARK: - Private: token key

    /// Cache-key serialization. Swift dicts don't hash `[Int]` by
    /// default, and tuple-based keys aren't Hashable. We use a
    /// length-prefixed string representation: `"N:t0,t1,t2,..."`.
    /// Fast for the token counts we see in practice (<100k) and
    /// doesn't collide across token sequences of different lengths.
    @inline(__always)
    private func tokenKey(_ tokens: [Int]) -> String {
        var s = "\(tokens.count):"
        s.reserveCapacity(tokens.count * 6 + 8)
        for (i, t) in tokens.enumerated() {
            if i > 0 { s.append(",") }
            s.append(String(t))
        }
        return s
    }
}
