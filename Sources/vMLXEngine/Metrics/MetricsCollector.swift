import Foundation
import MLX

/// Live performance metrics collector — GPU memory, RAM, CPU, tokens/sec,
/// queue depth, latency. Feeds the Server screen's PerformancePanel and any
/// future /admin/perf endpoint.
///
/// Design notes:
/// - Single long-lived polling Task wakes every `pollInterval` and fan-outs a
///   `Snapshot` to all subscribers.
/// - Token throughput is a rolling window (last 5s): we keep (timestamp, count)
///   tuples in ring buffers and compute rate = sum / windowSeconds on each
///   snapshot. Separate buffers for prefill vs decode.
/// - GPU memory comes from `MLX.Memory.snapshot()` (activeMemory + cacheMemory
///   + peakMemory). This is the vmlx-swift-lm fork's canonical API — verified
///   at .build/checkouts/mlx-swift/Source/MLX/Memory.swift.
/// - RAM uses mach_task_basic_info for resident size (process RSS). Total RAM
///   from ProcessInfo.processInfo.physicalMemory.
/// - CPU% uses host_cpu_load_info deltas between snapshots (user+system+nice).
public actor MetricsCollector {

    // MARK: Snapshot

    public struct Snapshot: Sendable {
        public let timestamp: Date
        public let gpuMemBytesUsed: Int64
        public let gpuMemBytesPeak: Int64
        public let ramBytesUsed: Int64
        public let ramBytesTotal: Int64
        public let cpuPercent: Double
        public let tokensPerSecondRolling: Double
        public let promptTokensPerSecondRolling: Double
        public let queueDepth: Int
        public let activeRequests: Int
        public let recentLatenciesMs: [Double]

        public init(
            timestamp: Date,
            gpuMemBytesUsed: Int64,
            gpuMemBytesPeak: Int64,
            ramBytesUsed: Int64,
            ramBytesTotal: Int64,
            cpuPercent: Double,
            tokensPerSecondRolling: Double,
            promptTokensPerSecondRolling: Double,
            queueDepth: Int,
            activeRequests: Int,
            recentLatenciesMs: [Double]
        ) {
            self.timestamp = timestamp
            self.gpuMemBytesUsed = gpuMemBytesUsed
            self.gpuMemBytesPeak = gpuMemBytesPeak
            self.ramBytesUsed = ramBytesUsed
            self.ramBytesTotal = ramBytesTotal
            self.cpuPercent = cpuPercent
            self.tokensPerSecondRolling = tokensPerSecondRolling
            self.promptTokensPerSecondRolling = promptTokensPerSecondRolling
            self.queueDepth = queueDepth
            self.activeRequests = activeRequests
            self.recentLatenciesMs = recentLatenciesMs
        }
    }

    // MARK: State

    private let pollInterval: TimeInterval
    private let windowSeconds: TimeInterval = 5.0
    private let latencyBufferMax = 20

    private var decodeSamples: [(Date, Int)] = []
    private var prefillSamples: [(Date, Int)] = []
    private var latencies: [Double] = []
    private var queueDepth: Int = 0
    private var activeRequests: Int = 0

    /// Monotone-increasing peak GPU memory (bytes). Surfaced via `Snapshot.gpuMemBytesPeak`.
    /// Separate from `MLX.Memory.snapshot().peakMemory` so tests (and the
    /// "Reset peak" button on the GPU tile) can drive it deterministically.
    private var peakGPUBytes: Int64 = 0
    /// Most-recent observed active GPU bytes. Used to seed `peakGPUBytes` on reset.
    private var lastGPUActiveBytes: Int64 = 0
    /// One-shot warning flag so a zeroed-out GPU probe only logs once per process.
    private var warnedZeroGPU: Bool = false

    private var continuations: [UUID: AsyncStream<Snapshot>.Continuation] = [:]
    private var pollTask: Task<Void, Never>?

    // CPU% deltas
    private var lastCPUTicks: (user: UInt32, system: UInt32, idle: UInt32, nice: UInt32)?

    public init(pollInterval: TimeInterval = 1.0) {
        self.pollInterval = pollInterval
    }

    // MARK: Subscribe

    /// Subscribe to live metrics snapshots. Starts the polling task lazily on
    /// first subscription. Terminates the stream when the subscriber cancels.
    public func subscribe() -> AsyncStream<Snapshot> {
        let (stream, continuation) = AsyncStream<Snapshot>.makeStream()
        let id = UUID()
        continuations[id] = continuation
        continuation.onTermination = { [weak self] _ in
            Task { await self?.removeContinuation(id) }
        }
        if pollTask == nil {
            startPolling()
        }
        // Yield an immediate snapshot so the UI paints without waiting.
        continuation.yield(makeSnapshot())
        return stream
    }

    private func removeContinuation(_ id: UUID) {
        continuations.removeValue(forKey: id)
        if continuations.isEmpty {
            pollTask?.cancel()
            pollTask = nil
        }
    }

    private func startPolling() {
        let interval = pollInterval
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tick()
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            }
        }
    }

    private func tick() {
        let snap = makeSnapshot()
        broadcast(snap)
    }

    private func broadcast(_ snap: Snapshot) {
        for (_, c) in continuations { c.yield(snap) }
    }

    // MARK: Recording hooks

    /// Record a batch of tokens processed over `durationMs`. For decode, this
    /// is typically a single-token step; for prefill, a chunked prompt block.
    public func recordTokenBatch(prefill: Bool, count: Int, durationMs: Double) {
        let now = Date()
        if prefill {
            prefillSamples.append((now, count))
            trimWindow(&prefillSamples, now: now)
        } else {
            decodeSamples.append((now, count))
            trimWindow(&decodeSamples, now: now)
        }
    }

    /// Record end-to-end request latency. Bounded deque (last 20).
    public func recordRequest(latencyMs: Double) {
        latencies.append(latencyMs)
        if latencies.count > latencyBufferMax {
            latencies.removeFirst(latencies.count - latencyBufferMax)
        }
    }

    public func setQueueDepth(_ depth: Int) {
        queueDepth = max(0, depth)
    }

    public func incrementActiveRequests() {
        activeRequests += 1
    }

    public func decrementActiveRequests() {
        activeRequests = max(0, activeRequests - 1)
    }

    /// Reset the displayed GPU peak to the current active usage. Wired to the
    /// "Reset peak" button on the GPU memory tile so the user can re-measure
    /// after unloading a model.
    public func resetPeakMemory() {
        peakGPUBytes = lastGPUActiveBytes
    }

    /// Immediately produce a snapshot (used by tests + first subscribe).
    public func currentSnapshot() -> Snapshot {
        makeSnapshot()
    }

    // MARK: Snapshot composition

    private func makeSnapshot() -> Snapshot {
        let now = Date()
        trimWindow(&decodeSamples, now: now)
        trimWindow(&prefillSamples, now: now)

        let decodeCount = decodeSamples.reduce(0) { $0 + $1.1 }
        let prefillCount = prefillSamples.reduce(0) { $0 + $1.1 }
        let decodeRate = Double(decodeCount) / windowSeconds
        let prefillRate = Double(prefillCount) / windowSeconds

        let gpu = gpuMemoryBytes()
        lastGPUActiveBytes = gpu.active
        // Peak is monotone-increasing within a session; the "Reset peak" button
        // is the only thing that can push it back down (via resetPeakMemory()).
        peakGPUBytes = max(peakGPUBytes, gpu.active, gpu.peak)

        let ram = processResidentBytes()
        let total = Int64(ProcessInfo.processInfo.physicalMemory)
        let cpu = cpuPercent()

        return Snapshot(
            timestamp: now,
            gpuMemBytesUsed: gpu.active,
            gpuMemBytesPeak: peakGPUBytes,
            ramBytesUsed: ram,
            ramBytesTotal: total,
            cpuPercent: cpu,
            tokensPerSecondRolling: decodeRate,
            promptTokensPerSecondRolling: prefillRate,
            queueDepth: queueDepth,
            activeRequests: activeRequests,
            recentLatenciesMs: latencies
        )
    }

    private func trimWindow(_ buf: inout [(Date, Int)], now: Date) {
        let cutoff = now.addingTimeInterval(-windowSeconds)
        while let first = buf.first, first.0 < cutoff {
            buf.removeFirst()
        }
    }

    // MARK: GPU memory (MLX)

    /// When true, skip `MLX.Memory.snapshot()` entirely. Set by test shims
    /// (or any environment where the Metal library isn't available) so the
    /// collector can still be exercised without crashing on a missing
    /// `default.metallib`. In production the engine never sets this.
    public static var disableGPUProbe: Bool = {
        // XCTest loads the bundle without a Metal default library; probing
        // would crash inside Cmlx. Detect and auto-disable.
        return ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }()

    private func gpuMemoryBytes() -> (active: Int64, peak: Int64) {
        if Self.disableGPUProbe { return (0, 0) }
        let snap = MLX.Memory.snapshot()
        // activeMemory = buffers actively held by MLX (weights + live tensors).
        // cacheMemory  = recyclable buffer pool. We surface active + cache as
        // "used" because that reflects actual wired allocation pressure.
        let used = Int64(snap.activeMemory + snap.cacheMemory)
        let peak = Int64(snap.peakMemory)
        if used == 0 && peak == 0 && !warnedZeroGPU {
            warnedZeroGPU = true
            // Metal device not initialized yet (no model loaded, no tensors
            // allocated). Surface once so ops can distinguish from a broken
            // probe. stderr keeps it out of user-visible logs.
            FileHandle.standardError.write(Data(
                "[MetricsCollector] MLX.Memory.snapshot() returned zero — Metal device not yet initialized; will self-heal on first allocation.\n".utf8
            ))
        }
        return (used, peak)
    }

    // MARK: Process RAM via mach_task_basic_info

    private func processResidentBytes() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Int64(info.resident_size)
    }

    // MARK: Host CPU%

    private func cpuPercent() -> Double {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        let user   = info.cpu_ticks.0
        let system = info.cpu_ticks.1
        let idle   = info.cpu_ticks.2
        let nice   = info.cpu_ticks.3

        defer { lastCPUTicks = (user, system, idle, nice) }
        guard let last = lastCPUTicks else { return 0 }

        let du = Double(user   &- last.user)
        let ds = Double(system &- last.system)
        let di = Double(idle   &- last.idle)
        let dn = Double(nice   &- last.nice)
        let busy = du + ds + dn
        let total = busy + di
        guard total > 0 else { return 0 }
        return (busy / total) * 100.0
    }
}
