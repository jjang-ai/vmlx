import Foundation

/// Thread-safe, bounded, multi-subscriber log store for the vMLX engine.
///
/// Mirrors the `logger = logging.getLogger("vmlx")` setup in
/// `vmlx_engine/server.py` plus the Electron-side stdout parser in
/// `panel/src/main/sessions.ts`. In the Swift rewrite we no longer fork a
/// Python sidecar, so the "log stream" is native: every engine component
/// that previously printed to stdout now calls `Engine.logs.append(...)`
/// and every subscriber (LogsPanel, CLI `--tail`, future file writer)
/// receives the same ordered stream.
///
/// ## Threading model
///
/// `LogStore` is an `actor`, so `append`, `subscribe`, `snapshot`, `clear`,
/// and `export` are all serialized on the actor's executor. `AsyncStream`
/// continuations are stored alongside their per-subscriber minimum level
/// in `[UUID: Subscriber]` and each `append` fan-outs to every registered
/// continuation whose level threshold is met. `AsyncStream.Continuation`
/// is documented safe to yield from any isolation domain; yielding from
/// inside the actor keeps delivery order deterministic with respect to
/// `append` call order.
///
/// New subscribers get an immediate replay of up to `replayCount`
/// most-recent buffered lines (matching the "replay recent" expectation
/// of the UI live-tail) then switch to live-tail on subsequent appends.
public actor LogStore {

    // MARK: - Types

    public struct Line: Sendable, Identifiable, Codable {
        public let id: UUID
        public let timestamp: Date
        public let level: Level
        public let category: String
        public let message: String

        public init(
            id: UUID = UUID(),
            timestamp: Date = Date(),
            level: Level,
            category: String,
            message: String
        ) {
            self.id = id
            self.timestamp = timestamp
            self.level = level
            self.category = category
            self.message = message
        }
    }

    public enum Level: String, Sendable, Comparable, CaseIterable, Codable {
        case trace, debug, info, warn, error

        private var rank: Int {
            switch self {
            case .trace: return 0
            case .debug: return 1
            case .info:  return 2
            case .warn:  return 3
            case .error: return 4
            }
        }
        public static func < (lhs: Level, rhs: Level) -> Bool { lhs.rank < rhs.rank }
    }

    private struct Subscriber {
        let continuation: AsyncStream<Line>.Continuation
        let minLevel: Level
    }

    // MARK: - Storage

    /// Hand-rolled circular buffer. We avoid pulling in swift-collections'
    /// `Deque` for this target — keeps the Swift rewrite dep footprint small,
    /// and the API surface we need (append + snapshot) is trivial.
    private struct Ring {
        private var items: [Line] = []
        let capacity: Int

        init(capacity: Int) {
            self.capacity = max(1, capacity)
            items.reserveCapacity(self.capacity)
        }

        mutating func append(_ line: Line) {
            if items.count >= capacity {
                items.removeFirst(items.count - capacity + 1)
            }
            items.append(line)
        }

        mutating func clear() { items.removeAll(keepingCapacity: true) }
        func all() -> [Line] { items }
    }

    private var ring: Ring
    private var subscribers: [UUID: Subscriber] = [:]

    /// Number of historic lines replayed to new subscribers.
    private let replayCount = 50

    public init(capacity: Int = 5000) {
        self.ring = Ring(capacity: capacity)
    }

    // MARK: - Writing

    public func append(_ level: Level, category: String, _ message: String) {
        let line = Line(level: level, category: category, message: message)
        ring.append(line)
        for (_, sub) in subscribers where line.level >= sub.minLevel {
            sub.continuation.yield(line)
        }
    }

    // MARK: - Reading

    public func snapshot(filter: LogFilter? = nil) -> [Line] {
        let all = ring.all()
        guard let f = filter else { return all }
        return all.filter(f.matches)
    }

    public func clear() {
        ring.clear()
    }

    /// Serialize the full buffer as NDJSON (one JSON object per line).
    public func export() -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        var out = Data()
        for line in ring.all() {
            if let data = try? encoder.encode(line) {
                out.append(data)
                out.append(0x0A) // \n
            }
        }
        return out
    }

    // MARK: - Subscription

    public nonisolated func subscribe(minLevel: Level = .info) -> AsyncStream<Line> {
        AsyncStream { continuation in
            let id = UUID()
            Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                await self.register(id: id, continuation: continuation, minLevel: minLevel)
            }
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                Task { await self.markAndUnregister(id: id) }
            }
        }
    }

    /// Set of subscribers that were torn down by their consumer before
    /// the register Task had a chance to run. Iter-29: without this,
    /// `onTermination` → `unregister` could land before `register`,
    /// leaving a phantom continuation in `subscribers` that every
    /// `append` would fan out to forever. `markAndUnregister` records
    /// the id here so `register` can refuse to add it in the first
    /// place on a late-arriving registration.
    private var tombstoned: Set<UUID> = []

    private func markAndUnregister(id: UUID) {
        tombstoned.insert(id)
        unregister(id: id)
        // Keep the tombstone set from growing unbounded — the vast
        // majority of tombstones fire BEFORE register and get consumed
        // there. Any that don't are already leaking memory per
        // pre-iter-29 behavior, so opportunistically prune.
        if tombstoned.count > 64 { tombstoned.removeAll() }
    }

    private func register(
        id: UUID,
        continuation: AsyncStream<Line>.Continuation,
        minLevel: Level
    ) {
        // Iter-29: if the consumer already terminated (tombstoned), the
        // register call arrived late — don't add the entry, just finish
        // the continuation so any replay lines don't get queued into a
        // stream nobody is reading.
        if tombstoned.remove(id) != nil {
            continuation.finish()
            return
        }
        // Replay last N lines that match the subscriber's threshold.
        let recent = ring.all().suffix(replayCount).filter { $0.level >= minLevel }
        for line in recent {
            continuation.yield(line)
        }
        subscribers[id] = Subscriber(continuation: continuation, minLevel: minLevel)
    }

    private func unregister(id: UUID) {
        if let sub = subscribers.removeValue(forKey: id) {
            sub.continuation.finish()
        }
    }
}

// MARK: - LogFilter

public struct LogFilter: Sendable {
    public var minLevel: LogStore.Level
    public var categories: Set<String>?
    public var contains: String?
    public var since: Date?

    public init(
        minLevel: LogStore.Level = .info,
        categories: Set<String>? = nil,
        contains: String? = nil,
        since: Date? = nil
    ) {
        self.minLevel = minLevel
        self.categories = categories
        self.contains = contains
        self.since = since
    }

    public func matches(_ line: LogStore.Line) -> Bool {
        if line.level < minLevel { return false }
        if let cats = categories, !cats.isEmpty, !cats.contains(line.category) { return false }
        if let s = contains, !s.isEmpty,
           !line.message.localizedCaseInsensitiveContains(s) { return false }
        if let since = since, line.timestamp < since { return false }
        return true
    }
}
