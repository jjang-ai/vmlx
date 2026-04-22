import SwiftUI
import vMLXEngine
import vMLXTheme

/// Live performance dashboard for the Server screen. Subscribes to
/// `Engine.subscribeMetrics()` and renders:
///   - 4 stat tiles: GPU memory (+ Reset peak), RAM, tokens/sec, Requests
///   - Decode tok/s sparkline (last 60 samples)
///   - Prefill tok/s sparkline (last 60 samples)
///   - Recent request latency bar chart (last 20 requests)
///
/// When `app.engineState == .stopped` the panel renders an empty state instead
/// of flashing zeros.
struct PerformancePanel: View {
    @Environment(AppState.self) private var app

    @State private var snapshot: MetricsCollector.Snapshot?
    @State private var decodeHistory: [Double] = []
    @State private var prefillHistory: [Double] = []
    /// R3 §304 — latest paged-cache hit rate (0..1) + hit/miss/evict
    /// counts. Polled every 2s from `Engine.cacheStats()` alongside
    /// the event-driven metrics stream.
    @State private var cacheHitRate: Double? = nil
    @State private var cacheHits: Int = 0
    @State private var cacheMisses: Int = 0
    private let sparklineCapacity = 60

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            Text("PERFORMANCE")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)

            if isStopped {
                emptyState
            } else {
                content
            }
        }
        .padding(Theme.Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.lg)
                .fill(Theme.Colors.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.lg)
                        .stroke(Theme.Colors.border, lineWidth: 1)
                )
        )
        .task(id: ObjectIdentifier(app.engine)) {
            for await snap in await app.engine.subscribeMetrics() {
                self.snapshot = snap
                append(&decodeHistory, snap.tokensPerSecondRolling)
                append(&prefillHistory, snap.promptTokensPerSecondRolling)
            }
        }
        // R3 §304 — cache-hit-rate poll (paged tier). 2s cadence so the
        // gauge tracks multi-turn T1→T2 hit changes without thrashing
        // the engine actor. Cancelled when the view is torn down.
        .task(id: ObjectIdentifier(app.engine)) {
            while !Task.isCancelled {
                if let stats = try? await app.engine.cacheStats(),
                   let paged = stats["paged"] as? [String: Any]
                {
                    let h = (paged["hits"] as? Int) ?? 0
                    let m = (paged["misses"] as? Int) ?? 0
                    let rate = (paged["hitRate"] as? Double)
                        ?? (h + m > 0 ? Double(h) / Double(h + m) : 0)
                    await MainActor.run {
                        self.cacheHitRate = rate
                        self.cacheHits = h
                        self.cacheMisses = m
                    }
                }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private var isStopped: Bool {
        if case .stopped = app.engineState { return true }
        return false
    }

    // MARK: - Content

    private var content: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            tiles
            sparklineSection(
                title: "Decode tokens/sec (last \(sparklineCapacity)s)",
                values: decodeHistory,
                color: Theme.Colors.accent
            )
            sparklineSection(
                title: "Prefill tokens/sec (last \(sparklineCapacity)s)",
                values: prefillHistory,
                color: Theme.Colors.accentHi
            )
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                Text("Recent request latency (ms)")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                latencyBars
                    .frame(height: 56)
                    .padding(Theme.Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.md)
                            .fill(Theme.Colors.surfaceHi)
                    )
            }
        }
    }

    private func sparklineSection(title: String, values: [Double], color: Color) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(title)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
            Sparkline(values: values, color: color)
                .frame(height: 56)
                .padding(Theme.Spacing.sm)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.md)
                        .fill(Theme.Colors.surfaceHi)
                )
        }
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Spacing.xs) {
            Image(systemName: "gauge.with.dots.needle.0percent")
                .font(.system(size: 28, weight: .regular))
                .foregroundStyle(Theme.Colors.textLow)
            Text("Engine stopped — no metrics")
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Colors.textLow)
        }
        .frame(maxWidth: .infinity, minHeight: 180, alignment: .center)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    // MARK: Tiles

    private var tiles: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible()),
        ], spacing: Theme.Spacing.sm) {
            gpuTile
            ramTile
            tokTile
            queueTile
            cacheHitTile
        }
    }

    /// R3 §304 — live paged-cache hit rate tile. Derived from
    /// `cacheStats().paged.hitRate`. Shows "—" before the first poll
    /// lands; fill-bar tracks the rate so users can see it climb on
    /// multi-turn reuse.
    private var cacheHitTile: some View {
        let rate = cacheHitRate ?? 0
        let totalReqs = cacheHits + cacheMisses
        return Tile(
            label: "Cache hit rate",
            value: cacheHitRate == nil
                ? "—"
                : String(format: "%.0f%%", rate * 100),
            valueFont: Theme.Typography.title,
            numeric: true
        ) {
            if totalReqs > 0 {
                Text("\(cacheHits)/\(totalReqs) turns hit")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            } else {
                Text("no cache activity yet")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
            }
        }
    }

    private var gpuTile: some View {
        let used = snapshot?.gpuMemBytesUsed ?? 0
        let peak = max(snapshot?.gpuMemBytesPeak ?? 0, used, Int64(1))
        let ratio = min(1.0, Double(used) / Double(peak))
        return Tile(
            label: "GPU Memory",
            value: "\(formatBytes(used)) / \(formatBytes(peak))",
            numeric: true,
            trailing: {
                AnyView(
                    Button {
                        Task { await app.engine.resetPeakMemory() }
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(Theme.Colors.textLow)
                    }
                    .buttonStyle(.plain)
                    .help("Reset peak to current")
                )
            }
        ) {
            FillBar(fraction: ratio, color: Theme.Colors.accent)
        }
    }

    private var ramTile: some View {
        let used = snapshot?.ramBytesUsed ?? 0
        let total = max(snapshot?.ramBytesTotal ?? 0, used, Int64(1))
        let ratio = min(1.0, Double(used) / Double(total))
        return Tile(
            label: "RAM (process)",
            value: "\(formatBytes(used)) / \(formatBytes(total))",
            numeric: true
        ) {
            FillBar(fraction: ratio, color: Theme.Colors.accentHi)
        }
    }

    private var tokTile: some View {
        let rate = snapshot?.tokensPerSecondRolling ?? 0
        return Tile(
            label: "Tokens/sec",
            value: formatNumber(rate, fractionDigits: 1),
            valueFont: Theme.Typography.title,
            numeric: true
        ) {
            Text("decoded (5s avg)")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    private var queueTile: some View {
        let active = snapshot?.activeRequests ?? 0
        let pending = snapshot?.queueDepth ?? 0
        return Tile(
            label: "Requests",
            value: "\(active) active",
            valueFont: Theme.Typography.title,
            numeric: true
        ) {
            Text("\(pending) queued")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Colors.textLow)
        }
    }

    // MARK: Latency bars

    private var latencyBars: some View {
        GeometryReader { geo in
            let latencies = snapshot?.recentLatenciesMs ?? []
            // Use p95 as the scale ceiling so a single outlier doesn't squash
            // the rest of the bars to 1px. Fall back to max for tiny samples.
            let scaleMax = latencyScaleMax(latencies)
            let n = max(latencies.count, 1)
            let barWidth = max(2, (geo.size.width / CGFloat(n)) - 2)
            HStack(alignment: .bottom, spacing: 2) {
                if latencies.isEmpty {
                    Text("No requests yet")
                        .font(Theme.Typography.caption)
                        .foregroundStyle(Theme.Colors.textLow)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    ForEach(Array(latencies.enumerated()), id: \.offset) { idx, v in
                        let clamped = min(v, scaleMax)
                        let h = CGFloat(clamped / scaleMax) * geo.size.height
                        let isOutlier = v > scaleMax
                        RoundedRectangle(cornerRadius: 1)
                            .fill((isOutlier ? Theme.Colors.accentHi : Theme.Colors.accent).opacity(0.85))
                            .frame(width: barWidth, height: max(2, h))
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                            .id(idx)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
            .animation(.easeOut(duration: 0.25), value: latencies)
        }
    }

    private func latencyScaleMax(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 1 }
        let sorted = values.sorted()
        let idx = max(0, min(sorted.count - 1, Int(Double(sorted.count) * 0.95)))
        let p95 = sorted[idx]
        return max(p95, 1)
    }

    // MARK: History helpers

    private func append(_ buf: inout [Double], _ value: Double) {
        buf.append(value)
        if buf.count > sparklineCapacity {
            buf.removeFirst(buf.count - sparklineCapacity)
        }
    }

    // MARK: Formatting

    private static let numberFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        f.maximumFractionDigits = 2
        return f
    }()

    private func formatNumber(_ v: Double, fractionDigits: Int = 0) -> String {
        let f = Self.numberFormatter
        f.minimumFractionDigits = fractionDigits
        f.maximumFractionDigits = fractionDigits
        return f.string(from: NSNumber(value: v)) ?? "0"
    }

    /// Byte formatter — always surfaces GB with 1 decimal once we're past MB so
    /// "12.4 GB" is readable. Smaller values use whole-number MB / KB.
    private func formatBytes(_ bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var idx = 0
        while value >= 1024 && idx < units.count - 1 {
            value /= 1024
            idx += 1
        }
        let digits = (idx >= 3) ? 1 : 0
        return String(format: "%.\(digits)f %@", value, units[idx])
    }
}

// MARK: - Private building blocks

private struct Tile<Accessory: View>: View {
    let label: String
    let value: String
    var valueFont: Font = Theme.Typography.bodyHi
    var numeric: Bool = false
    var trailing: (() -> AnyView)? = nil
    @ViewBuilder let accessory: () -> Accessory

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Text(label)
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Colors.textLow)
                Spacer(minLength: 0)
                trailing?()
            }
            valueLabel
            accessory()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.md)
                .fill(Theme.Colors.surfaceHi)
        )
    }

    @ViewBuilder
    private var valueLabel: some View {
        let text = Text(value)
            .font(valueFont)
            .foregroundStyle(Theme.Colors.textHigh)
            .lineLimit(1)
            .minimumScaleFactor(0.7)
        if numeric {
            text
                .contentTransition(.numericText())
                .animation(.easeOut(duration: 0.2), value: value)
        } else {
            text
        }
    }
}

private struct FillBar: View {
    let fraction: Double
    let color: Color
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Theme.Colors.border)
                RoundedRectangle(cornerRadius: 2)
                    .fill(color)
                    .frame(width: max(0, geo.size.width * CGFloat(fraction)))
                    .animation(.easeOut(duration: 0.25), value: fraction)
            }
        }
        .frame(height: 4)
    }
}
