import Foundation

/// Idle lifecycle timer. Fires `.softSleep` after `softAfter` seconds of
/// inactivity, and `.deepSleep` after `deepAfter` seconds (absolute — measured
/// from the same `lastActivity` anchor as soft, NOT relative to the soft event).
///
/// Mirrors the Python `vmlx_engine/server.py` idle watcher: `admin_soft_sleep`
/// (line ~1880) fires after ~5 min of no `/v1/chat/completions` traffic, and
/// `admin_deep_sleep` (line ~1927) follows ~10 min later. The Swift rewrite
/// drops the HTTP round-trip and wires the timer directly into the `Engine`
/// actor.
///
/// ## Threading model
///
/// `IdleTimer` is an `actor`. A single long-lived background `Task` (spawned
/// in `init`) samples wall-clock elapsed-since-last-activity once per second
/// and fans out to subscribers. All mutable state lives on the actor's
/// executor; subscribers receive events on an `AsyncStream`.
///
/// ## Cancellation
///
/// `deinit` cancels the loop Task explicitly. Because the loop is inside a
/// `Task { [weak self] ... }`, once `self` is released the loop exits
/// naturally on the next tick, but we also call `cancel()` so sleep returns
/// promptly. Subscribers are `finish()`ed on deinit so downstream
/// `for await` loops terminate.
///
/// ## Edge cases
///
/// - If `enabled == false` the loop still runs but never broadcasts — toggling
///   enabled back on resumes from the current lastActivity (no backfill).
/// - Once `.softSleep` fires, state is marked `softFired` and won't fire again
///   until `reset()` is called.
/// - `.deepSleep` fires only after `.softSleep` has fired (strict order).
/// - `reset()` clears both fired flags and refreshes lastActivity.
public actor IdleTimer {

    public struct Config: Sendable, Equatable {
        public var softAfter: TimeInterval
        public var deepAfter: TimeInterval
        public var enabled: Bool

        public init(
            softAfter: TimeInterval = 300,
            deepAfter: TimeInterval = 900,
            enabled: Bool = true
        ) {
            self.softAfter = softAfter
            self.deepAfter = deepAfter
            self.enabled = enabled
        }
    }

    public enum Event: Sendable, Equatable {
        case softSleep
        case deepSleep
    }

    // MARK: - State

    private var config: Config
    private var lastActivity: Date = Date()
    private var softFired: Bool = false
    private var deepFired: Bool = false
    private var subscribers: [UUID: AsyncStream<Event>.Continuation] = [:]
    private var loopTask: Task<Void, Never>?

    /// Poll interval for the sampling loop. Exposed for tests via
    /// `init(config:tickInterval:)`.
    private let tickInterval: TimeInterval

    public init(config: Config = .init(), tickInterval: TimeInterval = 0.05) {
        self.config = config
        self.tickInterval = tickInterval
        Task { await self.startLoop() }
    }

    deinit {
        loopTask?.cancel()
        for (_, cont) in subscribers {
            cont.finish()
        }
    }

    // MARK: - Public API

    public func reset() {
        lastActivity = Date()
        softFired = false
        deepFired = false
    }

    public func setConfig(_ config: Config) {
        self.config = config
        // Toggling config does not reset lastActivity — if the user shortens
        // softAfter while already idle, the next tick may fire immediately.
    }

    public func currentConfig() -> Config { config }

    public func subscribe() -> AsyncStream<Event> {
        AsyncStream { continuation in
            let id = UUID()
            self.register(id: id, continuation: continuation)
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                Task { await self.unregister(id: id) }
            }
        }
    }

    // MARK: - Internals

    private func register(id: UUID, continuation: AsyncStream<Event>.Continuation) {
        subscribers[id] = continuation
    }

    private func unregister(id: UUID) {
        if let c = subscribers.removeValue(forKey: id) {
            c.finish()
        }
    }

    private func broadcast(_ event: Event) {
        for (_, cont) in subscribers {
            cont.yield(event)
        }
    }

    private func startLoop() {
        loopTask?.cancel()
        loopTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tick()
                let nanos = await self.tickNanos()
                try? await Task.sleep(nanoseconds: nanos)
            }
        }
    }

    private func tickNanos() -> UInt64 {
        UInt64(max(0.001, tickInterval) * 1_000_000_000)
    }

    private func tick() {
        guard config.enabled else { return }
        let elapsed = Date().timeIntervalSince(lastActivity)
        if !softFired, elapsed >= config.softAfter {
            softFired = true
            broadcast(.softSleep)
        }
        if softFired, !deepFired, elapsed >= config.deepAfter {
            deepFired = true
            broadcast(.deepSleep)
        }
    }
}
