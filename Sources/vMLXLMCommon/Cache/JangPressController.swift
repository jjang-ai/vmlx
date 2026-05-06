// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressController — failsafe idle-time compression driver for
// the routed-expert weight tier.
//
// SAFETY MODEL (v1)
// =================
// The controller NEVER touches a tile while inference is in flight.
// The only window it engages is when ALL of these are true:
//
//   • No active generation request (engine reports idle)
//   • Last decode tick was > `quiesceTimeoutMs` ms ago
//   • At least one of:
//       - Memory pressure WARNING or CRITICAL received, OR
//       - User explicitly requested compaction, OR
//       - App entered background mode (NSApplication willResignActive)
//
// When the next inference request arrives, the controller flips ALL
// volatile tiles back to non-volatile BEFORE the engine touches them.
// The first token may be ~50-200 ms slower (kernel decompresses) but
// correctness is identical to never having compressed.
//
// FAILURE MODES
// =============
// • Tile DISCARDED by kernel under critical pressure → refault from
//   disk via `JangPressMachCache.acquire`. Slower (~ms-tens
//   of ms per tile) but lossless.
// • Cache disabled mid-flight → wake handler hit before engine reads;
//   no observable effect.
// • Engine-side bug that calls inference without going through the
//   wake hook → tiles still resident (failsafe: kernel only compresses
//   under explicit pressure, and only after our quiesce timeout, so
//   races are rare).
//
// FUTURE (v2)
// ===========
// Per-decode-step volatility flips on actively dormant experts. Gated
// on an MLX-swift fork that exposes `MTLBuffer.setPurgeableState` per
// tensor. Not in this version.

import Foundation
import Dispatch
import os

public enum JangPressState: Sendable {
    case disabled       // controller off — no tiles compressed
    case armed          // controller on, all tiles non-volatile (active)
    case quiescing      // controller on, ticks counting down to compress
    case compressed     // controller on, cold tiles volatile
}

public protocol JangPressObserver: AnyObject {
    func emberDidEnterState(_ state: JangPressState)
}

public final class JangPressController: @unchecked Sendable {

    /// Mach VM backend (.mach). Holds its own copy of weights in
    /// purgeable VM regions; kernel does WKdm compression. Optional —
    /// at most one of `cache` / `mmapTier` is non-nil.
    private let cache: JangPressMachCache?

    /// File-backed mmap backend (.mmap). Reads bundle safetensors via
    /// PROT_READ; release calls madvise(DONTNEED), forceRelease calls
    /// msync(MS_INVALIDATE) for guaranteed reclaim. Optional —
    /// mutually exclusive with `cache`.
    private let mmapTier: JangPressMmapTier?

    private let log = Logger(subsystem: "ai.jangq.vmlx", category: "JangPress")
    private let lock = OSAllocatedUnfairLock()

    /// Time (ms) after the last inference activity before the controller
    /// considers the engine "quiet enough to compress". Default 30 s.
    private let quiesceTimeoutMs: Int

    /// Tracked routing frequency per (layer, expert). Used to pick which
    /// tiles are "cold" enough to compress when armed.
    private var routingFreq: [TileKey: UInt64] = [:]
    private var totalRoutes: UInt64 = 0
    private var skippedRouteObservations: UInt64 = 0
    private var canonicalWillNeedBytes: UInt64 = 0
    private var canonicalDontNeedBytes: UInt64 = 0

    /// Fraction of tiles to keep non-volatile even when compressed. The
    /// hottest `keepHotFraction` of routed experts (by frequency) stay
    /// resident at all times.
    private let keepHotFraction: Double

    /// Is the engine actively generating right now? Set by the engine
    /// before each inference and cleared after.
    private var inferenceInFlight: Bool = false

    /// iter 24: tracks whether we've completed the first inference.
    /// On the first transition from in-flight → finished, we trigger
    /// the post-load page-cache reclaim — by then MLX has finished
    /// pread'ing weights into Metal buffers and the kernel page cache
    /// is pure redundancy.
    private var firstInferenceCompleted: Bool = false

    /// Wall-clock of last inference tick. Used to determine quiesce.
    private var lastInferenceTick: Date = Date()

    /// State machine — written under `lock`.
    private var state: JangPressState = .disabled

    private var pressureSource: DispatchSourceMemoryPressure?
    private var quiesceTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "ai.jangq.vmlx.jang-press", qos: .utility)

    private weak var observer: JangPressObserver?

    fileprivate struct TileKey: Hashable {
        let layer: Int
        let expert: Int
    }

    /// When true (default), `compressColdTiles` uses
    /// `mmapTier.forceRelease` (msync MS_INVALIDATE) which forces
    /// the kernel to drop pages immediately. Aggressive — empirically
    /// 7.7 GB reclaim on DSV4-Flash, but 3× decode slowdown because
    /// MLX's reads also share those pages and refault from disk.
    ///
    /// When false ("soft-only"), uses `mmapTier.release`
    /// (madvise MADV_DONTNEED) which is a HINT — kernel ignores
    /// under low pressure, acts on it under high pressure. No
    /// slowdown when memory is roomy; full reclaim when it isn't.
    /// **Recommended for production.**
    private let useForceRelease: Bool

    /// False when MLX's canonical weights are already mmap-backed.
    /// In that mode we avoid the auxiliary tier's hard `msync` path and
    /// use `JangPressCanonicalMmapAdvisor` for soft router-aware advice
    /// against the exact canonical MLX mmap records instead.
    private let explicitPageReclaimEnabled: Bool

    /// Whether decode-time router telemetry should drive eager WILLNEED
    /// prefetch through this controller. Canonical mmap mode leaves this
    /// false by default: router-aware canonical advice has its own opt-in
    /// coordinator, and paying synchronous top-k readback for the auxiliary
    /// mmap tier regresses Laguna-class MoE decode throughput.
    public let routePrefetchEnabled: Bool

    /// Init with the .mach backend.
    public init(
        cache: JangPressMachCache,
        quiesceTimeoutMs: Int = 30_000,
        keepHotFraction: Double = 0.30,
        routePrefetchEnabled: Bool = true,
        observer: JangPressObserver? = nil
    ) {
        self.cache = cache
        self.mmapTier = nil
        self.quiesceTimeoutMs = quiesceTimeoutMs
        self.keepHotFraction = keepHotFraction
        self.useForceRelease = true       // .mach uses Mach VOLATILE — no msync involved
        self.explicitPageReclaimEnabled = true
        self.routePrefetchEnabled = routePrefetchEnabled
        self.observer = observer
    }

    /// Init with the .mmap backend. Same lifecycle (arm /
    /// willStartInference / didFinishInference / compressColdTiles)
    /// but compaction strategy depends on `useForceRelease`:
    /// `true` → msync MS_INVALIDATE (hard reclaim, slows decode);
    /// `false` → madvise MADV_DONTNEED (soft hint, failsafe).
    public init(
        mmapTier: JangPressMmapTier,
        quiesceTimeoutMs: Int = 30_000,
        keepHotFraction: Double = 0.30,
        useForceRelease: Bool = false,
        explicitPageReclaimEnabled: Bool = true,
        routePrefetchEnabled: Bool = true,
        observer: JangPressObserver? = nil
    ) {
        self.cache = nil
        self.mmapTier = mmapTier
        self.useForceRelease = useForceRelease
        self.explicitPageReclaimEnabled = explicitPageReclaimEnabled
        self.routePrefetchEnabled = routePrefetchEnabled
        self.quiesceTimeoutMs = quiesceTimeoutMs
        self.keepHotFraction = keepHotFraction
        self.observer = observer
    }

    deinit {
        pressureSource?.cancel()
        quiesceTimer?.cancel()
    }

    // MARK: - Lifecycle

    /// Arm the controller — start watching for idle + memory pressure.
    /// All tiles begin non-volatile (full speed).
    public func arm() {
        lock.withLock {
            guard state == .disabled else { return }
            state = .armed
            installPressureSource()
        }
        log.notice("armed (quiesceTimeout=\(self.quiesceTimeoutMs) ms, keepHot=\(self.keepHotFraction))")
        notifyObserver()
    }

    /// Disarm — restore all tiles to non-volatile and stop watching.
    public func disarm() {
        lock.withLock {
            guard state != .disabled else { return }
            state = .disabled
            pressureSource?.cancel()
            pressureSource = nil
            quiesceTimer?.cancel()
            quiesceTimer = nil
        }
        // Wake all tiles before disarm (failsafe).
        wakeAll()
        log.notice("disarmed (all tiles non-volatile)")
        notifyObserver()
    }

    // MARK: - Engine integration hooks

    /// Engine calls this BEFORE every inference. Wakes any compressed
    /// tiles back to non-volatile (kernel decompresses on access). MUST
    /// be called or the engine may see a tile mid-decompress.
    public func willStartInference(layerExpertHints: [(layer: Int, experts: [Int])] = []) {
        let needsWake = lock.withLock { () -> Bool in
            inferenceInFlight = true
            lastInferenceTick = Date()
            quiesceTimer?.cancel()
            quiesceTimer = nil
            return state == .compressed
        }
        if needsWake {
            log.notice("inference incoming → waking all volatile tiles")
            wakeAll(hints: layerExpertHints)
            lock.withLock { state = .armed }
            notifyObserver()
        }
    }

    /// Engine calls this AFTER inference completes (or aborts). Starts
    /// the quiesce countdown if armed. iter 24: also triggers the
    /// post-load page-cache reclaim on the FIRST call.
    public func didFinishInference() {
        let shouldFirstReclaim: Bool = lock.withLock {
            inferenceInFlight = false
            lastInferenceTick = Date()
            if state == .armed || state == .quiescing {
                state = .quiescing
                scheduleQuiesce()
            }
            let first = !firstInferenceCompleted
            firstInferenceCompleted = true
            return first
        }

        // iter 24: AFTER MLX has finished pread'ing weights into Metal
        // buffers (which definitely happened by the end of the first
        // inference), the kernel page cache holding those same bytes
        // is pure redundancy. Drop it asynchronously off the inference
        // path. This is the v1 "real win" of JangPress on JANGTQ
        // bundles where MLX always copies.
        if shouldFirstReclaim, explicitPageReclaimEnabled, let tier = mmapTier {
            queue.async { [weak self] in
                let t0 = Date()
                tier.releaseAllRoutedRanges()
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                let s = tier.snapshotIfBuilt()
                let routedMB = s.totalRoutedBytes / 1024 / 1024
                let msg = "[JangPressController] first-inference reclaim: madvise DONTNEED on \(s.expertCount) tiles (\(routedMB) MB routed) in \(ms) ms\n"
                FileHandle.standardError.write(Data(msg.utf8))
                self?.log.notice("first-inference reclaim: madvise DONTNEED on \(s.expertCount) tiles (\(routedMB) MB routed) in \(ms) ms")
                self?.notifyObserver()
            }
        }
    }

    /// Engine calls this on every router decision so we can track
    /// routing frequency per expert. Cheap (single dictionary update
    /// under the lock).
    public func recordRoute(layer: Int, experts: [Int]) {
        lock.withLock {
            for e in experts {
                routingFreq[TileKey(layer: layer, expert: e), default: 0] &+= 1
                totalRoutes &+= 1
            }
        }
    }

    /// iter 27: batch-record routing observations for many (layer, experts)
    /// pairs in ONE lock acquisition. Designed for model-side opt-in
    /// wiring where the model collects per-token routing decisions
    /// across an entire forward pass and flushes once at the end. Per-
    /// token `recordRoute` calls would acquire the lock once per token
    /// per layer (40 layers × 128 tokens = 5120 lock-acquires per turn);
    /// this is one acquisition for the whole batch.
    ///
    /// `pairs`: list of (layer, experts) tuples, one per (token × layer)
    /// pair the model observed. Order doesn't matter; only frequency.
    public func recordBatchRoutes(_ pairs: [(layer: Int, experts: [Int])]) {
        var uniqueKeys = Set<TileKey>()
        for (layer, experts) in pairs {
            for e in Set(experts) where layer >= 0 && e >= 0 {
                uniqueKeys.insert(TileKey(layer: layer, expert: e))
            }
        }
        var prefetchByLayer: [Int: [Int]] = [:]
        for key in uniqueKeys {
            prefetchByLayer[key.layer, default: []].append(key.expert)
        }
        lock.withLock {
            for (layer, experts) in pairs {
                for e in experts {
                    routingFreq[TileKey(layer: layer, expert: e), default: 0] &+= 1
                    totalRoutes &+= 1
                }
            }
        }
        // Router-aware warm path. `recordTopK` is called immediately after
        // the gate selects experts and before the SwitchMLP/TurboQuantSwitch
        // dispatch reads their weights. On mmap-backed bundles this issues
        // MADV_WILLNEED for the selected ranges so cold pages can fault in
        // before the expert matmul touches them. The call is cheap/no-op on
        // dense or non-mmap models.
        if routePrefetchEnabled {
            for (layer, experts) in prefetchByLayer {
                mmapTier?.acquire(layer: layer, experts: experts)
            }
            adviseCanonicalMmap(.willNeed, keys: Array(uniqueKeys))
        }
    }

    public func recordSkippedRouteObservation() {
        lock.withLock {
            skippedRouteObservations &+= 1
        }
    }

    /// iter 27: probe whether any routing frequency has been observed
    /// since the controller was armed. False means model hasn't opted
    /// into recordRoute / recordBatchRoutes — `compressColdTiles` will
    /// fall back to uniform release.
    public var hasRoutingObservations: Bool {
        lock.withLock { !routingFreq.isEmpty }
    }

    /// Manual user-driven compaction (e.g. from a "free up RAM" UI
    /// button). Only fires if armed and idle.
    public func manualCompact() {
        let armed = lock.withLock { state == .armed || state == .quiescing }
        guard armed else { return }
        log.notice("manual compaction requested")
        compressColdTiles()
    }

    // MARK: - Internals

    private func installPressureSource() {
        let src = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: queue)
        src.setEventHandler { [weak self] in
            guard let self else { return }
            let event = src.data
            var labels: [String] = []
            if event.contains(.warning) { labels.append("warning") }
            if event.contains(.critical) { labels.append("critical") }
            let eventLabel = labels.isEmpty ? "unknown" : labels.joined(separator: "|")
            self.log.notice("memory pressure event: \(eventLabel)")
            // SAFETY: never compress while inference is in flight. Even
            // soft DONTNEED can cause page-cache evictions that affect
            // MLX's reads of the same files. Only fire when we're
            // demonstrably between requests.
            //
            // We also require state ∈ {armed, quiescing} — disabled and
            // compressed states already mean we're either off or in
            // the right place.
            let canCompress = self.lock.withLock { () -> Bool in
                guard !self.inferenceInFlight else { return false }
                return self.state == .armed || self.state == .quiescing
            }
            if canCompress {
                self.compressColdTiles()
            } else {
                let snapshot = self.lock.withLock {
                    (inFlight: self.inferenceInFlight,
                     state: JangPressController.stateLabel(self.state))
                }
                self.log.notice("pressure event ignored: inferenceInFlight=\(snapshot.inFlight) state=\(snapshot.state)")
            }
        }
        src.activate()
        pressureSource = src
    }

    private static func stateLabel(_ state: JangPressState) -> String {
        switch state {
        case .disabled: return "disabled"
        case .armed: return "armed"
        case .quiescing: return "quiescing"
        case .compressed: return "compressed"
        }
    }

    private func scheduleQuiesce() {
        quiesceTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + .milliseconds(quiesceTimeoutMs))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let stillIdle = self.lock.withLock {
                !self.inferenceInFlight && self.state == .quiescing
            }
            if stillIdle {
                self.log.notice("quiesce timeout reached → compressing cold tiles")
                self.compressColdTiles()
            }
        }
        timer.resume()
        quiesceTimer = timer
    }

    private func compressColdTiles() {
        guard explicitPageReclaimEnabled else {
            compressCanonicalMmapColdTiles()
            return
        }

        // Pick which tiles count as "cold": the bottom (1 - keepHot)
        // fraction by routing frequency. Tiles never seen go to cold.
        let snapshot = lock.withLock { (Array(routingFreq), Set(routingFreq.keys), totalRoutes) }
        let (freq, _, _) = snapshot

        // Iter 27: when routingFreq is empty (no model has called
        // `recordRoute()` — which is the default since per-token
        // hooking is opt-in per model family), the old per-tile sort
        // produced zero work + the quiesce-timer pathway was a no-op.
        // Fall back to releasing EVERY routed range — same effect as
        // iter 24's first-inference reclaim. Conservative: drops the
        // entire kernel page-cache reflection of the routed mass.
        // The "keep hot 30%" semantic is forfeited (uniform release)
        // but on a v1 .mmap path that's tracking page-cache redundancy
        // anyway, this is functionally identical for users.
        if freq.isEmpty {
            log.notice("compressColdTiles: empty routingFreq → uniform release (no opt-in recordRoute observed)")
            if useForceRelease {
                cache?.release(layer: -1, experts: [])     // .mach is no-op for layer=-1
                mmapTier?.forceReleaseAllRoutedRanges()
            } else {
                mmapTier?.releaseAllRoutedRanges()
            }
            lock.withLock { state = .compressed }
            notifyObserver()
            return
        }

        // Sort tiles by frequency ascending — lowest = coldest
        let sorted = freq.sorted { $0.value < $1.value }
        let coldCount = Int(Double(sorted.count) * (1.0 - keepHotFraction))
        let cold = sorted.prefix(coldCount).map { $0.key }

        // Iter 143 — wire `JangPressMachCache.pinHot`. Previously zero
        // callers; under memory pressure the kernel could WKdm-compress
        // hot expert tiles (the top-K most routed) and stall the next
        // decode on a refault. Pinning them as VM_PURGABLE_NONVOLATILE
        // tells the kernel "never compress" — mid-decode stalls
        // disappear. Hot set = sort.dropFirst(coldCount). pinHot is
        // idempotent (Set-based) so re-firing on subsequent compress
        // passes only adds new entries when the hot/cold partition
        // shifts.
        if let cache = cache {
            let hot = sorted.dropFirst(coldCount).map { $0.key }
            var hotByLayer: [Int: [Int]] = [:]
            for k in hot { hotByLayer[k.layer, default: []].append(k.expert) }
            for (layer, experts) in hotByLayer where !experts.isEmpty {
                cache.pinHot(layer: layer, experts: experts)
            }
        }

        // Group by layer for the per-backend release call shape.
        var byLayer: [Int: [Int]] = [:]
        for k in cold { byLayer[k.layer, default: []].append(k.expert) }
        for (layer, experts) in byLayer {
            // .mach backend: kernel honors VOLATILE flag for WKdm
            //                compression; soft release is enough.
            // .mmap backend: madvise(DONTNEED) is hint-only on macOS;
            //                use forceRelease (msync MS_INVALIDATE) for
            //                guaranteed reclaim. We're past the
            //                quiesce timeout so the latency cost on
            //                next acquire is acceptable.
            cache?.release(layer: layer, experts: experts)
            if useForceRelease {
                mmapTier?.forceRelease(layer: layer, experts: experts)
            } else {
                mmapTier?.release(layer: layer, experts: experts)
            }
        }
        let total = byLayer.values.reduce(0) { $0 + $1.count }
        log.notice("compressed \(total) cold tiles (across \(byLayer.count) layers, kept \(self.keepHotFraction * 100, format: .fixed(precision: 0)) % hot)")
        lock.withLock { state = .compressed }
        notifyObserver()
    }

    /// Canonical mmap compaction path. This is deliberately soft-only:
    /// it sends MADV_DONTNEED to the exact MLX no-copy safetensors ranges
    /// while the engine is idle, but never uses `msync(MS_INVALIDATE)` on
    /// buffers Metal may refault on the next decode. Router observations
    /// make this frequency-aware; otherwise we fall back to the configured
    /// cold percentage over all routed canonical records.
    private func compressCanonicalMmapColdTiles() {
        let snapshot = lock.withLock { Array(routingFreq) }
        let coldPercent = max(
            0,
            min(100, Int(((1.0 - keepHotFraction) * 100.0).rounded())))

        if snapshot.isEmpty {
            guard coldPercent > 0 else {
                lock.withLock { state = .armed }
                notifyObserver()
                return
            }
            let bytes = JangPressCanonicalMmapAdvisor.adviseRouted(
                .dontNeed,
                percent: coldPercent)
            if bytes > 0 {
                lock.withLock {
                    canonicalDontNeedBytes &+= UInt64(bytes)
                    state = .compressed
                }
                log.notice("canonical mmap compressed \(bytes) routed bytes with uniform \(coldPercent)% DONTNEED")
            } else {
                lock.withLock { state = .armed }
                log.notice("canonical mmap compaction found no registered routed mmap records")
            }
            notifyObserver()
            return
        }

        let sorted = snapshot.sorted { $0.value < $1.value }
        let coldCount = Int(Double(sorted.count) * (1.0 - keepHotFraction))
        let cold = sorted.prefix(coldCount).map { $0.key }
        let bytes = adviseCanonicalMmap(.dontNeed, keys: Array(cold))
        lock.withLock { state = bytes > 0 ? .compressed : .armed }
        log.notice("canonical mmap compressed \(bytes) cold bytes from \(cold.count) frequency-ranked tiles")
        notifyObserver()
    }

    private func wakeAll(hints: [(layer: Int, experts: [Int])] = []) {
        // For every tile we ever compressed, flip non-volatile. We can't
        // easily enumerate every registered tile from here, so the
        // engine's hint list is the fast path; otherwise we acquire by
        // walking the routing-frequency dict (every tile we've routed
        // at least once is in there).
        if !hints.isEmpty {
            var keys: [TileKey] = []
            for h in hints {
                _ = try? cache?.acquire(layer: h.layer, experts: h.experts)
                mmapTier?.acquire(layer: h.layer, experts: h.experts)
                for e in h.experts where h.layer >= 0 && e >= 0 {
                    keys.append(TileKey(layer: h.layer, expert: e))
                }
            }
            adviseCanonicalMmap(.willNeed, keys: keys)
            return
        }
        let allKeys = lock.withLock { Array(routingFreq.keys) }
        if allKeys.isEmpty, !explicitPageReclaimEnabled {
            let bytes = JangPressCanonicalMmapAdvisor.adviseRouted(.willNeed, percent: 100)
            if bytes > 0 {
                lock.withLock { canonicalWillNeedBytes &+= UInt64(bytes) }
            }
            return
        }
        var byLayer: [Int: [Int]] = [:]
        for k in allKeys { byLayer[k.layer, default: []].append(k.expert) }
        for (layer, experts) in byLayer {
            _ = try? cache?.acquire(layer: layer, experts: experts)
            mmapTier?.acquire(layer: layer, experts: experts)
        }
        adviseCanonicalMmap(.willNeed, keys: allKeys)
    }

    @discardableResult
    private func adviseCanonicalMmap(
        _ advice: JangPressAdvice,
        keys: [TileKey]
    ) -> Int64 {
        guard !explicitPageReclaimEnabled, !keys.isEmpty else { return 0 }
        let unique = Array(Set(keys))
        let bytes = JangPressCanonicalMmapAdvisor.adviseExperts(
            advice,
            pairs: unique.map { (layer: $0.layer, expert: $0.expert) })
        guard bytes > 0 else { return bytes }
        lock.withLock {
            switch advice {
            case .willNeed:
                canonicalWillNeedBytes &+= UInt64(bytes)
            case .dontNeed:
                canonicalDontNeedBytes &+= UInt64(bytes)
            case .sequential, .random:
                break
            }
        }
        return bytes
    }

    private func notifyObserver() {
        guard let obs = observer else { return }
        let s = lock.withLock { state }
        DispatchQueue.main.async { obs.emberDidEnterState(s) }
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var state: JangPressState
        public var inferenceInFlight: Bool
        public var lastInferenceMsAgo: Int
        public var totalRoutesObserved: UInt64
        public var distinctTilesObserved: Int
        public var keepHotFraction: Double
        public var firstInferenceCompleted: Bool
        public var compactionMode: String   // "frequency-aware" | "uniform-fallback"
        public var explicitPageReclaimEnabled: Bool
        public var canonicalAdviceEnabled: Bool
        public var canonicalWillNeedBytes: UInt64
        public var canonicalDontNeedBytes: UInt64
        public var skippedRouteObservations: UInt64
        public var routePrefetchEnabled: Bool
    }

    public func snapshot() -> Stats {
        lock.withLock {
            Stats(
                state: state,
                inferenceInFlight: inferenceInFlight,
                lastInferenceMsAgo: Int(Date().timeIntervalSince(lastInferenceTick) * 1000),
                totalRoutesObserved: totalRoutes,
                distinctTilesObserved: routingFreq.count,
                keepHotFraction: keepHotFraction,
                firstInferenceCompleted: firstInferenceCompleted,
                compactionMode: routingFreq.isEmpty ? "uniform-fallback" : "frequency-aware",
                explicitPageReclaimEnabled: explicitPageReclaimEnabled,
                canonicalAdviceEnabled: !explicitPageReclaimEnabled,
                canonicalWillNeedBytes: canonicalWillNeedBytes,
                canonicalDontNeedBytes: canonicalDontNeedBytes,
                skippedRouteObservations: skippedRouteObservations,
                routePrefetchEnabled: routePrefetchEnabled
            )
        }
    }
}
