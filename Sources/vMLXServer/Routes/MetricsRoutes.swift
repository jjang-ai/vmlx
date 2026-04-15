import Foundation
import HTTPTypes
import Hummingbird
import vMLXEngine

/// Prometheus-format `/metrics` endpoint.
///
/// Emits one text/plain response per scrape. Metric names follow the
/// Prometheus best-practices naming guide (snake_case, unit suffix,
/// base units — seconds not ms, bytes not MB). Types are explicit
/// (`# TYPE <name> <counter|gauge|histogram>`) so client libraries
/// don't have to guess.
///
/// Scope: every metric we surface comes from `MetricsCollector`, the
/// engine's `LogStore` (for error counts) and a tiny shared counter for
/// request-level accounting. If vMLX.Server is bound to a single Engine
/// the output is that engine's view; the gateway variant enumerates
/// every registered engine and sums / reports per-session breakdowns.
public enum MetricsRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        router.get("/metrics") { _, _ -> Response in
            let text = await render(engine: engine)
            var headers: HTTPFields = [:]
            headers[.contentType] = "text/plain; version=0.0.4; charset=utf-8"
            return Response(
                status: .ok,
                headers: headers,
                body: .init(byteBuffer: .init(string: text))
            )
        }
    }

    /// Gateway-facing `/metrics` that enumerates every registered engine.
    /// Separate from the per-session handler above because each engine
    /// has its own MetricsCollector snapshot — we don't try to merge,
    /// we emit per-session labels.
    public static func registerGateway<Context: RequestContext>(
        on router: Router<Context>,
        enumerate: @Sendable @escaping () async -> [Engine]
    ) {
        router.get("/metrics") { _, _ -> Response in
            let engines = await enumerate()
            let text = await renderGateway(engines: engines)
            var headers: HTTPFields = [:]
            headers[.contentType] = "text/plain; version=0.0.4; charset=utf-8"
            return Response(
                status: .ok,
                headers: headers,
                body: .init(byteBuffer: .init(string: text))
            )
        }
    }

    // MARK: - Rendering

    private static func render(engine: Engine) async -> String {
        let snap = await engine.metrics.currentSnapshot()
        let errors = await engine.logs.snapshot().filter { $0.level == .error }.count
        return renderSnapshot(snap, errors: errors, sessionLabel: nil)
    }

    private static func renderGateway(engines: [Engine]) async -> String {
        var buffer = ""
        buffer += "# HELP vmlx_sessions Number of engines currently registered with the gateway.\n"
        buffer += "# TYPE vmlx_sessions gauge\n"
        buffer += "vmlx_sessions \(engines.count)\n\n"
        for (idx, engine) in engines.enumerated() {
            let snap = await engine.metrics.currentSnapshot()
            let errors = await engine.logs.snapshot().filter { $0.level == .error }.count
            let label = "session_\(idx)"
            buffer += renderSnapshot(snap, errors: errors, sessionLabel: label)
        }
        return buffer
    }

    /// Formats a single `Snapshot` as Prometheus text. When `sessionLabel`
    /// is non-nil, every metric line gets a `{session="<label>"}` label
    /// set so the gateway output distinguishes concurrent engines.
    private static func renderSnapshot(
        _ snap: MetricsCollector.Snapshot,
        errors: Int,
        sessionLabel: String?
    ) -> String {
        var buffer = ""
        let lbl = sessionLabel.map { "{session=\"\($0)\"}" } ?? ""

        // GPU memory.
        buffer += "# HELP vmlx_gpu_memory_bytes Active GPU memory used by MLX, in bytes.\n"
        buffer += "# TYPE vmlx_gpu_memory_bytes gauge\n"
        buffer += "vmlx_gpu_memory_bytes\(lbl) \(snap.gpuMemBytesUsed)\n"
        buffer += "# HELP vmlx_gpu_memory_peak_bytes Peak GPU memory observed since reset, in bytes.\n"
        buffer += "# TYPE vmlx_gpu_memory_peak_bytes gauge\n"
        buffer += "vmlx_gpu_memory_peak_bytes\(lbl) \(snap.gpuMemBytesPeak)\n\n"

        // RAM.
        buffer += "# HELP vmlx_ram_bytes_used Process resident set size, in bytes.\n"
        buffer += "# TYPE vmlx_ram_bytes_used gauge\n"
        buffer += "vmlx_ram_bytes_used\(lbl) \(snap.ramBytesUsed)\n"
        buffer += "# HELP vmlx_ram_bytes_total Total physical RAM on this host, in bytes.\n"
        buffer += "# TYPE vmlx_ram_bytes_total gauge\n"
        buffer += "vmlx_ram_bytes_total\(lbl) \(snap.ramBytesTotal)\n\n"

        // CPU %.
        buffer += "# HELP vmlx_cpu_usage_ratio Normalized CPU usage (0..1).\n"
        buffer += "# TYPE vmlx_cpu_usage_ratio gauge\n"
        buffer += "vmlx_cpu_usage_ratio\(lbl) \(snap.cpuPercent / 100.0)\n\n"

        // Throughput.
        buffer += "# HELP vmlx_decode_tokens_per_second Rolling 5s decode token throughput.\n"
        buffer += "# TYPE vmlx_decode_tokens_per_second gauge\n"
        buffer += "vmlx_decode_tokens_per_second\(lbl) \(snap.tokensPerSecondRolling)\n"
        buffer += "# HELP vmlx_prefill_tokens_per_second Rolling 5s prefill token throughput.\n"
        buffer += "# TYPE vmlx_prefill_tokens_per_second gauge\n"
        buffer += "vmlx_prefill_tokens_per_second\(lbl) \(snap.promptTokensPerSecondRolling)\n\n"

        // Queue + inflight.
        buffer += "# HELP vmlx_queue_depth Requests waiting in the scheduler queue.\n"
        buffer += "# TYPE vmlx_queue_depth gauge\n"
        buffer += "vmlx_queue_depth\(lbl) \(snap.queueDepth)\n"
        buffer += "# HELP vmlx_active_requests Requests currently generating tokens.\n"
        buffer += "# TYPE vmlx_active_requests gauge\n"
        buffer += "vmlx_active_requests\(lbl) \(snap.activeRequests)\n\n"

        // Errors.
        buffer += "# HELP vmlx_errors_total Total ERROR-level log events since process start.\n"
        buffer += "# TYPE vmlx_errors_total counter\n"
        buffer += "vmlx_errors_total\(lbl) \(errors)\n\n"

        // Request latency histogram. MetricsCollector keeps the last N
        // request latencies (in ms); we bucket them into a Prometheus
        // histogram on the fly. The bucket boundaries match what
        // Grafana's "chat completion latency" dashboard expects.
        let buckets: [Double] = [
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ]
        let seconds = snap.recentLatenciesMs.map { $0 / 1000.0 }
        var counts = [Int](repeating: 0, count: buckets.count)
        var cumulative = 0
        for s in seconds {
            for (i, le) in buckets.enumerated() where s <= le {
                counts[i] += 1
            }
        }
        buffer += "# HELP vmlx_request_duration_seconds Chat request latency histogram.\n"
        buffer += "# TYPE vmlx_request_duration_seconds histogram\n"
        for (i, le) in buckets.enumerated() {
            cumulative = counts[i]
            buffer += "vmlx_request_duration_seconds_bucket\(labelJoin(lbl, extra: "le=\"\(le)\"")) \(cumulative)\n"
        }
        buffer += "vmlx_request_duration_seconds_bucket\(labelJoin(lbl, extra: "le=\"+Inf\"")) \(seconds.count)\n"
        let sum = seconds.reduce(0, +)
        buffer += "vmlx_request_duration_seconds_sum\(lbl) \(sum)\n"
        buffer += "vmlx_request_duration_seconds_count\(lbl) \(seconds.count)\n\n"

        return buffer
    }

    /// Merges a bare label set with an extra `key="value"` pair so the
    /// callers don't have to manually handle the empty-vs-populated
    /// label-set distinction.
    private static func labelJoin(_ existing: String, extra: String) -> String {
        if existing.isEmpty { return "{\(extra)}" }
        // Drop trailing `}` and re-add with the extra field appended.
        let trimmed = existing.dropLast()
        return "\(trimmed),\(extra)}"
    }
}
