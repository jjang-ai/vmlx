// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressCanonicalExpertAdvisor — router-aware advisory coordinator
// for the canonical MLX safetensors mmap registry.
//
// Why this is separate from `JangPressController`:
//   • The controller is a state machine over compress/wake transitions
//     and quiesce timers. It treats every observed (layer, expert) the
//     same and forwards to the auxiliary mmap tier.
//   • This advisor is a per-layer hot-set tracker. It observes top-k
//     router decisions, keeps the most-recently-seen experts warm with
//     MADV_WILLNEED, and ages out the rest with MADV_DONTNEED on the
//     SAME canonical no-copy mmap ranges that mlx-swift hands to Metal.
//
// The OS still owns actual page reclaim. This is precise hot/cold page
// advice — not a custom compressed blob format.
//
// Adapted from the upstream osaurus reference at
// `vmlx-swift-lm/Libraries/MLXLMCommon/Cache/JangPressCanonicalExpertAdvisor.swift`.
// Differences from the reference:
//   • Drops the `JangPressLoadOptions` parameter (no such struct in
//     vmlx-swift) — `configure(...)` takes explicit params from
//     `Engine.LoadOptions`.
//   • Drops dlsym indirection — `mlx_safetensors_mmap_advise_experts`
//     is exported directly through the local Cmlx target.
//   • Accepts `VMLX_JANGPRESS_DEBUG` in addition to the upstream
//     `JANGPRESS_DEBUG` for env-naming consistency with the rest of the
//     vmlx codebase.

import Cmlx
import Foundation
import MLX

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct JangPressCanonicalExpertAdvisorStatus: Sendable, Equatable {
    public var enabled: Bool
    public var asyncReadback: Bool
    public var warmAdvice: Bool
    public var hotPerLayer: Int
    public var hotExpertCount: Int
    public var warmCalls: Int
    public var coldCalls: Int
    public var warmBytes: Int64
    public var coldBytes: Int64
    public var pendingObservations: Int
    public var droppedQueueFull: Int
    public var skippedLargeIndexTensors: Int
    public var skippedTracerArrays: Int
    public var readbacks: Int
    /// Number of times an expert that was previously cold-advised (LRU
    /// evicted) is observed back in a router top-k decision and has to
    /// be re-promoted to the hot set. This is the canonical thrash
    /// signal: high `rewarms` relative to `coldCalls` means the current
    /// `compressPct` / `hotPerLayer` budget is too aggressive for the
    /// workload's expert reuse pattern. Operators tuning a model can
    /// use the ratio `rewarms / max(1, coldCalls)` as a "we evicted
    /// pages we shouldn't have" rate. Canonical mmap pages stay in the
    /// kernel page cache opportunistically (MADV_DONTNEED is a hint),
    /// so a high rate doesn't always mean a hard refault — but it does
    /// mean the advisor's policy is fighting the router.
    public var rewarms: Int
    /// Number of distinct (layer, expert) pairs that have ever been
    /// cold-advised in the current configID's lifetime. Bounded by
    /// `numLayers × numRoutedExperts`. Surfacing this lets the user
    /// see how broad the eviction footprint has been independent of
    /// total `coldCalls` (which counts repeated evictions).
    public var distinctColdAdvisedPairs: Int
}

public final class JangPressCanonicalExpertAdvisor: @unchecked Sendable {
    public static let shared = JangPressCanonicalExpertAdvisor()

    private struct Config {
        var enabled: Bool = false
        var asyncReadback: Bool = true
        var warmAdvice: Bool = false
        var hotPerLayer: Int = 32
        var maxIndicesPerReadback: Int = 32
        var maxPendingObservations: Int = 512
        var drainBatchSize: Int = 64
        var debug: Bool = false
    }

    private final class PendingObservation: @unchecked Sendable {
        let configID: UInt64
        let layer: Int
        let expertIDs: [Int32]

        init(configID: UInt64, layer: Int, expertIDs: [Int32]) {
            self.configID = configID
            self.layer = layer
            self.expertIDs = expertIDs
        }
    }

    private struct MutableState {
        var config = Config()
        var configID: UInt64 = 0
        var generation: UInt64 = 0
        var hotByLayer: [Int: [Int: UInt64]] = [:]
        /// Per-layer set of expert IDs that have been cold-advised at
        /// least once during this configID's lifetime. Used to detect
        /// rewarms: when a router top-k surfaces an expert that's both
        /// (a) absent from the hot set and (b) present in this set, we
        /// know it was previously evicted and is now coming back.
        var coldHistoryByLayer: [Int: Set<Int>] = [:]
        var pendingObservations: [PendingObservation] = []
        var workerScheduled = false
        var warmCalls = 0
        var coldCalls = 0
        var warmBytes: Int64 = 0
        var coldBytes: Int64 = 0
        var droppedQueueFull = 0
        var skippedLargeIndexTensors = 0
        var skippedTracerArrays = 0
        var readbacks = 0
        var rewarms = 0
    }

    private let lock = NSLock()
    private let workerQueue = DispatchQueue(
        label: "ai.jangq.vmlx.jangpress.router-advisor",
        qos: .utility)
    private var state = MutableState()

    private init() {}

    /// Reconfigure the advisor for a fresh `Engine.load`. Resets the
    /// per-layer hot set and any in-flight pending observations from a
    /// prior model. Safe to call repeatedly; subsequent observations
    /// after a configID bump are silently dropped if they target the
    /// previous load.
    ///
    /// Caller responsibilities:
    ///   • `mmapEnabled` — true only when the Engine.load path actually
    ///     enabled `VMLX_MMAP_SAFETENSORS=1` (canonical no-copy
    ///     storage). Without that, the advice symbol's registry is
    ///     empty and every call returns 0 bytes.
    ///   • `enabled` — true only when JangPress is on AND backend is
    ///     `.mmap`. The advisor is meaningless for `.mach` (Mach VM
    ///     allocates fresh purgeable regions, no mmap to advise).
    ///   • `enableRouterAdvice` — opt-in. The current Engine.LoadOptions
    ///     does not carry this flag; pass false and rely on
    ///     `JANGPRESS_ROUTER_ADVICE=1` to switch on. Future load options
    ///     can plumb a real boolean through here.
    ///   • `compressPct` — used to derive a reasonable
    ///     `hotPerLayer` default when the env override is absent. Higher
    ///     compressPct → smaller hot set → more cold advice.
    public func configure(
        enabled: Bool,
        mmapEnabled: Bool,
        enableRouterAdvice: Bool,
        enableWarmAdvice: Bool = false,
        compressPct: Int,
        numRoutedExperts: Int? = nil,
        topK: Int? = nil
    ) {
        let env = ProcessInfo.processInfo.environment
        let routerEnv = env["JANGPRESS_ROUTER_ADVICE"]?.lowercased()
        let envEnabled = routerEnv == "1" || routerEnv == "true"
            || routerEnv == "on" || routerEnv == "yes"
        let envDisabled = routerEnv == "0" || routerEnv == "false"
            || routerEnv == "off" || routerEnv == "no"
        let resolvedEnabled = mmapEnabled
            && enabled
            && (enableRouterAdvice || envEnabled)
            && !envDisabled
        let hotPerLayer = parsePositiveEnv(
            "JANGPRESS_ROUTER_HOT_PER_LAYER",
            env: env,
            default: hotPerLayerDefault(
                compressPct: compressPct,
                numRoutedExperts: numRoutedExperts,
                topK: topK))
        let maxIndices = parsePositiveEnv(
            "JANGPRESS_ROUTER_MAX_INDICES",
            env: env,
            default: 32)
        let asyncReadback = parseBoolEnv(
            "JANGPRESS_ROUTER_ASYNC_READBACK",
            env: env,
            default: true)
        let warmAdvice = parseBoolEnv(
            "JANGPRESS_ROUTER_WARM_ADVICE",
            env: env,
            default: enableWarmAdvice)
        let maxPending = parsePositiveEnv(
            "JANGPRESS_ROUTER_MAX_PENDING",
            env: env,
            default: 512)
        let drainBatch = parsePositiveEnv(
            "JANGPRESS_ROUTER_DRAIN_BATCH",
            env: env,
            default: 64)
        let debug = env["JANGPRESS_ROUTER_DEBUG"] == "1"
            || env["VMLX_JANGPRESS_DEBUG"] == "1"
            || env["JANGPRESS_DEBUG"] == "1"

        lock.lock()
        state.configID &+= 1
        state.config = Config(
            enabled: resolvedEnabled,
            asyncReadback: asyncReadback,
            warmAdvice: warmAdvice,
            hotPerLayer: max(1, hotPerLayer),
            maxIndicesPerReadback: max(1, maxIndices),
            maxPendingObservations: max(1, maxPending),
            drainBatchSize: max(1, drainBatch),
            debug: debug)
        state.hotByLayer.removeAll(keepingCapacity: true)
        state.coldHistoryByLayer.removeAll(keepingCapacity: true)
        state.pendingObservations.removeAll(keepingCapacity: true)
        // Reset accumulators so per-load /health stats are not polluted
        // by a previous load. configID bump alone would not zero these.
        state.warmCalls = 0
        state.coldCalls = 0
        state.warmBytes = 0
        state.coldBytes = 0
        state.droppedQueueFull = 0
        state.skippedLargeIndexTensors = 0
        state.skippedTracerArrays = 0
        state.readbacks = 0
        state.rewarms = 0
        lock.unlock()

        if debug {
            FileHandle.standardError.write(Data(
                "[JangPressRouter] configure enabled=\(resolvedEnabled) async=\(asyncReadback) warmAdvice=\(warmAdvice) hotPerLayer=\(hotPerLayer) maxIndices=\(maxIndices) maxPending=\(maxPending) compressPct=\(compressPct)\n".utf8))
        }
    }

    /// Observe an MoE router top-k indices tensor for one layer.
    ///
    /// Tracer-array guard: MLX compile traces forwards with non-realized
    /// arrays. Calling `asArray` on a tracer SIGSEGVs in `MLXArray.asArray`
    /// (verified on Laguna compiled-decode). We bail out cleanly and
    /// count the skip; the compiled graph still routes correctly, only
    /// the advisory readback is skipped for that trace pass.
    public func observe(layer: Int, indices: MLXArray) {
        // Cheap pre-check: skip the lock + tracer probe entirely when
        // the advisor is disabled. Hot-path on dense or non-MoE models.
        lock.lock()
        let config = state.config
        let configID = state.configID
        if !config.enabled {
            lock.unlock()
            return
        }
        if indices.size > config.maxIndicesPerReadback {
            state.skippedLargeIndexTensors += 1
            lock.unlock()
            return
        }
        lock.unlock()

        var isTracer = false
        if mlx_array_is_tracer(&isTracer, indices.ctx) == 0, isTracer {
            lock.lock()
            state.skippedTracerArrays += 1
            lock.unlock()
            return
        }

        // Realize int32 indices on the caller's queue. Doing this on
        // the worker queue is not a stable MLX contract and has tickled
        // races during concurrent decode in the upstream advisor too.
        let uniqueExperts = readUniqueExperts(indices)
        guard !uniqueExperts.isEmpty else { return }

        if config.asyncReadback {
            enqueue(configID: configID, layer: layer, experts: uniqueExperts)
        } else {
            processExperts(
                configID: configID, layer: layer, uniqueExperts: uniqueExperts)
        }
    }

    /// Bypass entry for callers that have already realized indices to
    /// `[Int32]` (e.g. `JangPressRouteTelemetry.recordTopK` already did
    /// the host read with its own tracer guard + size cap).
    public func observe(layer: Int, experts: [Int32]) {
        lock.lock()
        let config = state.config
        let configID = state.configID
        if !config.enabled {
            lock.unlock()
            return
        }
        if experts.count > config.maxIndicesPerReadback {
            state.skippedLargeIndexTensors += 1
            lock.unlock()
            return
        }
        lock.unlock()

        let unique = Array(Set(experts.filter { $0 >= 0 })).sorted()
        guard !unique.isEmpty else { return }

        if config.asyncReadback {
            enqueue(configID: configID, layer: layer, experts: unique)
        } else {
            processExperts(
                configID: configID, layer: layer, uniqueExperts: unique)
        }
    }

    private func enqueue(
        configID: UInt64, layer: Int, experts: [Int32]
    ) {
        lock.lock()
        guard state.config.enabled, state.configID == configID else {
            lock.unlock()
            return
        }
        if state.pendingObservations.count >= state.config.maxPendingObservations {
            state.droppedQueueFull += 1
            lock.unlock()
            return
        }
        state.pendingObservations.append(PendingObservation(
            configID: configID, layer: layer, expertIDs: experts))
        let shouldSchedule = !state.workerScheduled
        if shouldSchedule {
            state.workerScheduled = true
        }
        lock.unlock()

        if shouldSchedule {
            workerQueue.async { [weak self] in
                self?.drainPendingObservations()
            }
        }
    }

    private func drainPendingObservations() {
        while true {
            lock.lock()
            let batchSize = max(1, state.config.drainBatchSize)
            if state.pendingObservations.isEmpty {
                state.workerScheduled = false
                lock.unlock()
                return
            }
            let count = min(batchSize, state.pendingObservations.count)
            let batch = Array(state.pendingObservations.prefix(count))
            state.pendingObservations.removeFirst(count)
            lock.unlock()

            for observation in batch {
                processExperts(
                    configID: observation.configID,
                    layer: observation.layer,
                    uniqueExperts: observation.expertIDs)
            }
        }
    }

    private func readUniqueExperts(_ indices: MLXArray) -> [Int32] {
        let routed = indices.reshaped([-1]).asType(.int32).asArray(Int32.self)
        return Array(Set(routed.filter { $0 >= 0 })).sorted()
    }

    private func processExperts(
        configID: UInt64,
        layer: Int,
        uniqueExperts: [Int32]
    ) {
        var warm: [(Int32, Int32)] = []
        var cold: [(Int32, Int32)] = []
        var debug = false
        var warmAdvice = false

        lock.lock()
        let config = state.config
        guard state.config.enabled, state.configID == configID else {
            lock.unlock()
            return
        }
        state.readbacks += 1
        state.generation &+= 1
        let generation = state.generation
        var hot = state.hotByLayer[layer] ?? [:]
        var coldHistory = state.coldHistoryByLayer[layer] ?? Set<Int>()
        var layerRewarms = 0
        for expert in uniqueExperts {
            let e = Int(expert)
            if hot[e] == nil {
                warm.append((Int32(layer), expert))
                // Rewarm detection: this expert is being newly added to
                // the hot set, AND it was previously cold-advised. The
                // OS may or may not have actually reclaimed the page —
                // we don't know — but the router is asking for it back,
                // which means our LRU policy was wrong about it being
                // cold. Drop from cold history so we don't double-count
                // if the expert oscillates in/out repeatedly.
                if coldHistory.remove(e) != nil {
                    layerRewarms &+= 1
                }
            }
            hot[e] = generation
        }

        if hot.count > config.hotPerLayer {
            let overflow = hot.count - config.hotPerLayer
            // LRU eviction by `generation`; ties broken by expert id
            // for determinism so two configs with identical traffic
            // produce identical advice traces.
            let evicted = hot
                .sorted { lhs, rhs in
                    if lhs.value == rhs.value { return lhs.key < rhs.key }
                    return lhs.value < rhs.value
                }
                .prefix(overflow)
            for (expert, _) in evicted {
                hot.removeValue(forKey: expert)
                cold.append((Int32(layer), Int32(expert)))
                coldHistory.insert(Int(expert))
            }
        }
        state.hotByLayer[layer] = hot
        state.coldHistoryByLayer[layer] = coldHistory
        state.rewarms &+= layerRewarms
        debug = config.debug
        warmAdvice = config.warmAdvice
        lock.unlock()

        if warmAdvice, !warm.isEmpty {
            let bytes = advise(pairs: warm, advice: 1)
            lock.lock()
            state.warmCalls += 1
            state.warmBytes += bytes
            lock.unlock()
        }
        if !cold.isEmpty {
            let bytes = advise(pairs: cold, advice: 0)
            lock.lock()
            state.coldCalls += 1
            state.coldBytes += bytes
            lock.unlock()
        }

        if debug, !warm.isEmpty || !cold.isEmpty {
            FileHandle.standardError.write(Data(
                "[JangPressRouter] layer=\(layer) warm=\(warm.count) cold=\(cold.count)\n".utf8))
        }
    }

    public func snapshot() -> JangPressCanonicalExpertAdvisorStatus {
        lock.lock()
        defer { lock.unlock() }
        let hotCount = state.hotByLayer.values.reduce(0) { $0 + $1.count }
        let coldHistoryCount = state.coldHistoryByLayer.values.reduce(0) {
            $0 + $1.count
        }
        return JangPressCanonicalExpertAdvisorStatus(
            enabled: state.config.enabled,
            asyncReadback: state.config.asyncReadback,
            warmAdvice: state.config.warmAdvice,
            hotPerLayer: state.config.hotPerLayer,
            hotExpertCount: hotCount,
            warmCalls: state.warmCalls,
            coldCalls: state.coldCalls,
            warmBytes: state.warmBytes,
            coldBytes: state.coldBytes,
            pendingObservations: state.pendingObservations.count,
            droppedQueueFull: state.droppedQueueFull,
            skippedLargeIndexTensors: state.skippedLargeIndexTensors,
            skippedTracerArrays: state.skippedTracerArrays,
            readbacks: state.readbacks,
            rewarms: state.rewarms,
            // distinctColdAdvisedPairs counts cold history entries that
            // are still tracked (i.e., not yet rewarmed back into hot).
            // To avoid making the metric ever-growing across rewarm
            // oscillations, we report the LIVE distinct cold footprint.
            // Total cumulative cold pairs is implicit in `coldCalls`.
            distinctColdAdvisedPairs: coldHistoryCount)
    }

    public func waitUntilIdle(timeoutSeconds: TimeInterval = 2.0) {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            lock.lock()
            let idle = state.pendingObservations.isEmpty
                && !state.workerScheduled
            lock.unlock()
            if idle { return }
            Thread.sleep(forTimeInterval: 0.005)
        }
    }

    /// Test-only: forcibly reset to disabled with cleared accumulators.
    /// Production callers go through `configure(...)` which preserves
    /// the same reset semantics on configID bump.
    internal func _testReset() {
        lock.lock()
        state.configID &+= 1
        state.config = Config()
        state.hotByLayer.removeAll(keepingCapacity: false)
        state.coldHistoryByLayer.removeAll(keepingCapacity: false)
        state.pendingObservations.removeAll(keepingCapacity: false)
        state.warmCalls = 0
        state.coldCalls = 0
        state.warmBytes = 0
        state.coldBytes = 0
        state.droppedQueueFull = 0
        state.skippedLargeIndexTensors = 0
        state.skippedTracerArrays = 0
        state.readbacks = 0
        state.rewarms = 0
        lock.unlock()
    }

    private func advise(pairs: [(Int32, Int32)], advice: Int32) -> Int64 {
        guard !pairs.isEmpty else { return 0 }
        let layers = pairs.map { $0.0 }
        let experts = pairs.map { $0.1 }
        return layers.withUnsafeBufferPointer { layerBuffer in
            experts.withUnsafeBufferPointer { expertBuffer in
                guard let lb = layerBuffer.baseAddress,
                      let eb = expertBuffer.baseAddress else { return 0 }
                return mlx_safetensors_mmap_advise_experts(
                    advice, lb, eb, Int64(pairs.count))
            }
        }
    }

    private func parsePositiveEnv(
        _ key: String,
        env: [String: String],
        default defaultValue: Int
    ) -> Int {
        guard let value = env[key], let parsed = Int(value), parsed > 0 else {
            return defaultValue
        }
        return parsed
    }

    private func parseBoolEnv(
        _ key: String,
        env: [String: String],
        default defaultValue: Bool
    ) -> Bool {
        guard let raw = env[key]?.lowercased() else {
            return defaultValue
        }
        if raw == "1" || raw == "true" || raw == "on" || raw == "yes" {
            return true
        }
        if raw == "0" || raw == "false" || raw == "off" || raw == "no" {
            return false
        }
        return defaultValue
    }

    /// Generic compressPct-only fallback. Used when caller did not pass
    /// `numRoutedExperts` (non-MoE bundle, sniff failed, or test). Same
    /// math as the upstream osaurus reference: `max(8, min(64, ⌈(1 -
    /// pct/100) · 128⌉))`. Saturates at 64 — that's safe for ≤ 64-expert
    /// architectures (Mixtral-8x, Phi-MoE) but starves 256-expert ones.
    private func defaultHotPerLayer(compressPct: Int) -> Int {
        let clamped = max(0, min(100, compressPct))
        let hotFraction = Double(100 - clamped) / 100.0
        return max(8, min(64, Int((hotFraction * 128.0).rounded(.up))))
    }

    /// Model-aware default. When `numRoutedExperts` is known we size
    /// the per-layer hot budget to keep the top-k working set resident
    /// even at high compressPct:
    ///
    ///     hot = clamp(
    ///         max(top_k · 4, ⌈N · (1 - pct/100)⌉),
    ///         lower = max(8, top_k · 2),
    ///         upper = N
    ///     )
    ///
    /// Why `top_k · 4`: a router that reuses experts across consecutive
    /// tokens (the common case for instruction-tuned MoEs) needs at
    /// least a few cycles' worth of expert IDs in the hot set before
    /// the LRU evicts something it will need again. Anything below
    /// `top_k · 2` thrashes for sure; `· 4` is the empirical knee.
    /// `topK == nil` falls back to the generic compressPct formula.
    private func hotPerLayerDefault(
        compressPct: Int,
        numRoutedExperts: Int?,
        topK: Int?
    ) -> Int {
        guard let n = numRoutedExperts, n > 0 else {
            return defaultHotPerLayer(compressPct: compressPct)
        }
        let pct = max(0, min(100, compressPct))
        let pctBudget = Int(ceil(Double(n) * Double(100 - pct) / 100.0))
        let topKFloor = max(1, topK ?? 4) * 4
        var hot = max(topKFloor, pctBudget)
        let lowerBound = max(8, max(1, topK ?? 4) * 2)
        if hot < lowerBound { hot = lowerBound }
        if hot > n { hot = n }
        return hot
    }
}
